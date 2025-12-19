"""
MouthMaster
25.12.15

Updated to support batch angle commands for vision-driven expression control.
Command format: "angles:A1,A2,...,A21" where A1-A21 are servo angles [0-180]
Servo order: JL,JR,LUL,LUR,LLL,LLR,CUL,CUR,CLL,CLR,TON,LR,UD,TL,BL,TR,BR,LO,LI,RI,RO
"""

import time
import sys
from machine import Pin
from servo import Servo


#####################################################
#              Servo Order Definition               #
#####################################################
# Order must match expression_control/protocol.py
SERVO_ORDER = [
    # Mouth (11): MouthMaster Pico - indices 0-10
    "JL", "JR", "LUL", "LUR", "LLL", "LLR",
    "CUL", "CUR", "CLL", "CLR", "TON",
    # Eyes (6): Eyes Pico - indices 11-16
    "LR", "UD", "TL", "BL", "TR", "BR",
    # Brows (4): Brows Pico - indices 17-20
    "LO", "LI", "RI", "RO"
]

# Indices for each subsystem
MOUTH_SERVO_INDICES = list(range(0, 11))   # 0-10
EYE_SERVO_INDICES = list(range(11, 17))    # 11-16
BROW_SERVO_INDICES = list(range(17, 21))   # 17-20

# Current angle state for all 21 servos (for readback)
current_angles = {name: 90 for name in SERVO_ORDER}

#####################################################
#                     GPIO                          #
#####################################################
led = Pin(25, Pin.OUT)                  # LED for debuging
mode = Pin(28, Pin.IN, Pin.PULL_UP)     # Mode 

# Eye control 
SDA = Pin(16, Pin.OUT)                  # Eye-Pico-16
SCL = Pin(17, Pin.OUT)                  # Eye-Pico-17 

# Brow control 
BSDA = Pin(18, Pin.OUT)                 # Brow-pico-16
BSCL = Pin(19, Pin.OUT)                 # Brow-pico-17 
#####################################################


#####################################################
#           Angle Command Parser                    #
#####################################################
def parse_angles_command(command):
    """
    Parse "angles:A1,A2,...,A21" format command.
    
    Args:
        command: String in format "angles:A1,A2,...,A21"
        
    Returns:
        List of 21 integer angles if valid, None otherwise
    """
    if not command.startswith("angles:"):
        return None
    
    angle_str = command[7:]  # Remove "angles:" prefix
    if not angle_str:
        return None
    
    parts = angle_str.split(",")
    if len(parts) != 21:
        return None
    
    angles = []
    for part in parts:
        try:
            angle = int(part.strip())
            if angle < 0 or angle > 180:
                return None
            angles.append(angle)
        except ValueError:
            return None
    
    return angles


def apply_angles(angles):
    """
    Apply 21 servo angles to the system.
    
    Args:
        angles: List of 21 integer angles [0-180]
                Order: JL,JR,LUL,LUR,LLL,LLR,CUL,CUR,CLL,CLR,TON,
                       LR,UD,TL,BL,TR,BR,LO,LI,RI,RO
    """
    global current_angles
    
    # Update current angles state
    for i, name in enumerate(SERVO_ORDER):
        current_angles[name] = angles[i]
    
    # Apply mouth servos (indices 0-10) directly
    for i in MOUTH_SERVO_INDICES:
        servo_name = SERVO_ORDER[i]
        if servo_name in servos:
            servos[servo_name].move(angles[i])
    
    # Translate eye angles (indices 11-16) to GPIO signals
    eye_angles = [angles[i] for i in EYE_SERVO_INDICES]
    translate_eye_angles_to_gpio(eye_angles)
    
    # Translate brow angles (indices 17-20) to GPIO signals
    brow_angles = [angles[i] for i in BROW_SERVO_INDICES]
    translate_brow_angles_to_gpio(brow_angles)


def translate_eye_angles_to_gpio(eye_angles):
    """
    Translate 6 eye servo angles to GPIO signals for Eyes Pico.
    
    GPIO patterns (SDA, SCL):
    - (0, 0): eyes_close - all servos to neutral/closed
    - (1, 0): eyes_open - all servos to open position
    - (1, 1): eyes_move - auto/movement mode
    
    Strategy: Determine eye state from angle values
    - If all angles near neutral (90): closed state
    - If eyelid angles indicate open: open state
    - Otherwise: movement mode (let Eyes Pico handle)
    
    Args:
        eye_angles: List of 6 angles [LR, UD, TL, BL, TR, BR]
    """
    # LR=0, UD=1, TL=2, BL=3, TR=4, BR=5
    lr, ud, tl, bl, tr, br = eye_angles
    
    # Check if eyes are in neutral/closed position
    # TL/TR near 90 and BL/BR near 90 indicates closed
    eyelid_angles = [tl, bl, tr, br]
    all_neutral = all(abs(a - 90) < 15 for a in eyelid_angles)
    
    # Check if eyelids are open (TL/BR high, BL/TR low based on limits)
    # From eyes.py: TL (90,160), BL (90,30), TR (90,30), BR (90,140)
    tl_open = tl > 120
    br_open = br > 120
    bl_open = bl < 60
    tr_open = tr < 60
    eyes_open = (tl_open or br_open) and (bl_open or tr_open)
    
    if all_neutral:
        # Closed state
        SDA.value(0)
        SCL.value(0)
    elif eyes_open:
        # Open state
        SDA.value(1)
        SCL.value(0)
    else:
        # Movement/auto mode - Eyes Pico will handle positioning
        SDA.value(1)
        SCL.value(1)


def translate_brow_angles_to_gpio(brow_angles):
    """
    Translate 4 brow servo angles to GPIO signals for Brows Pico.
    
    GPIO patterns (BSDA, BSCL):
    - (0, 0): brows_down - all brows in down position
    - (1, 1): brows_up - all brows raised
    - (1, 0): brows_angry - inner brows down, outer up
    - (0, 1): brows_happy - inner brows up, outer down
    
    From Brows.py servo limits:
    - LO: (80, 100) -> 80=down, 120=up
    - LI: (100, 80) -> 100=down, 60=up (inverted)
    - RI: (80, 100) -> 80=down, 120=up
    - RO: (100, 80) -> 100=down, 60=up (inverted)
    
    Args:
        brow_angles: List of 4 angles [LO, LI, RI, RO]
    """
    lo, li, ri, ro = brow_angles
    
    # Determine brow positions
    # Outer brows (LO, RO): higher angle = up
    lo_up = lo > 100
    ro_up = ro < 80  # Inverted
    
    # Inner brows (LI, RI): 
    li_up = li < 80  # Inverted
    ri_up = ri > 100
    
    outer_up = lo_up or ro_up
    inner_up = li_up or ri_up
    outer_down = not outer_up
    inner_down = not inner_up
    
    if outer_up and inner_up:
        # All brows up
        BSDA.value(1)
        BSCL.value(1)
    elif outer_down and inner_down:
        # All brows down
        BSDA.value(0)
        BSCL.value(0)
    elif outer_up and inner_down:
        # Angry expression
        BSDA.value(1)
        BSCL.value(0)
    elif outer_down and inner_up:
        # Happy expression
        BSDA.value(0)
        BSCL.value(1)
    else:
        # Default to down
        BSDA.value(0)
        BSCL.value(0)


def get_angles_readback():
    """
    Get current servo angles as a command string for data collection.
    
    Returns:
        String in format "angles:A1,A2,...,A21"
    """
    angle_values = [str(current_angles[name]) for name in SERVO_ORDER]
    return "angles:" + ",".join(angle_values)
#####################################################



#####################################################
#                ServoConfig Class                  # 
#####################################################
class SCFG:
    def __init__(self, pin_id, limits):
        self.servo = Servo(pin_id=pin_id)
        self.start, self.end = limits
        self.angle = self.start
        self.direction = 1  # Forward initially

    def move(self, angle):
        self.servo.write(angle)

    def step(self, step_size=1):
        # Calculate current step direction based on start vs end
        step = step_size if self.start < self.end else -step_size
        self.angle += self.direction * step

        # Reverse direction if past bounds
        if (self.direction == 1 and (
                (step > 0 and self.angle >= self.end) or (step < 0 and self.angle <= self.end))):
            self.angle = self.end
            self.direction = -1
        elif (self.direction == -1 and (
                (step > 0 and self.angle <= self.start) or (step < 0 and self.angle >= self.start))):
            self.angle = self.start
            self.direction = 1

        self.move(self.angle)

#####################################################



#####################################################
#                Servo Setup                        # 
#####################################################
servos = {
    "JL": SCFG(4, (90,50)),    # Jaw
    "JR": SCFG(5, (90,130)),
    
    "LUL": SCFG(6, (80,100)),  # Lip
    "LUR": SCFG(7, (80,100)),
    "LLL": SCFG(8, (10,170)),
    "LLR": SCFG(9, (80,100)),

    # i'm not sure what "C" stands for 
    # cheeks? corner of mouth? 
    "CUL": SCFG(10, (80,100)), 
    "CUR": SCFG(11, (80,100)),
    "CLL": SCFG(12, (80,100)),
    "CLR": SCFG(13, (80,100)),
    # ---------------------------------

    "TON": SCFG(14, (80,100)), # Tongue
    "EXA": SCFG(15, (80,100)), 
}

poses = {
    "mouth_closed": {
        "JL": 90, "JR": 90
    },
    "mouth_open": {
        "JL": 50, "JR": 130
    },
    "lips_down": {
        "LUL": 110, "LUR": 70, "LLL": 55, "LLR": 110
    },
    "lips_up": {
        "LUL": 40, "LUR": 140, "LLL": 110, "LLR": 60
    },
    "tongue_down": {
        "TON": 130
    },
    "tongue_up": {
        "TON": 55
    },
    "smile": {
        "CUL": 60, "CUR": 105, "CLL": 90, "CLR": 90
    },
    "frown": {
        "CUL": 100, "CUR": 75, "CLL": 90, "CLR": 90
    },    
    "wide": {
        "CUL": 65, "CUR": 105, "CLL": 120, "CLR": 65
    },
    "narrow": {
        "CUL": 90, "CUR": 80, "CLL": 90, "CLR": 90
    },
}

#####################################################
#               Apply Pose Function                 #
#####################################################
def apply_pose(pose_name):
    if pose_name in poses:
        for servo_name, angle in poses[pose_name].items():
            servos[servo_name].move(angle)
    else:
        print(f"Pose {pose_name} not found!")        



#####################################################
#                    Initialize                     #
#####################################################
print("Ready to receive commands")

SDA.value(0)
SCL.value(0)
BSDA.value(0)
BSCL.value(0)
apply_pose("mouth_closed")
apply_pose("tongue_down")
apply_pose("narrow")



#####################################################
#                    Main Loop                      #
#####################################################
while True:

    try:
        # Read command from USB (command from a PC)
        line = sys.stdin.readline()
        if not line:
            continue
        command = line.strip()
        print(f"Received: {command}")
        
        # LED
        if command == "on":
            led.value(1)
        elif command == "off":
            led.value(0)
        
        # Batch angle command (new format for vision control)
        elif command.startswith("angles:"):
            angles = parse_angles_command(command)
            if angles is not None:
                apply_angles(angles)
                print("OK")
            else:
                print("Error: Invalid angles command format")
        
        # Angle readback for data collection
        elif command == "get_angles":
            print(get_angles_readback())
        
        # Eyes
        elif command == "eyes_move": 
            SDA.value(1)
            SCL.value(1)
        elif command == "eyes_open": 
            SDA.value(1)
            SCL.value(0)
        elif command == "eyes_close": 
            SDA.value(0)
            SCL.value(0)
        
        # Brows
        elif command == "brows_up": 
            BSDA.value(1)
            BSCL.value(1)
        elif command == "brows_down": 
            BSDA.value(0)
            BSCL.value(0)
        elif command == "brows_happy": 
            BSDA.value(0)
            BSCL.value(1)
        elif command == "brows_angry": 
            BSDA.value(1)
            BSCL.value(0)
        
        # Mouth
        elif command == "mouth_closed":
            apply_pose("mouth_closed")
        elif command == "mouth_open":
            apply_pose("mouth_open")
        elif command == "lips_down":
            apply_pose("lips_down")
        elif command == "lips_up":
            apply_pose("lips_up")
        elif command == "tongue_down":
            apply_pose("tongue_down")
        elif command == "tongue_up":
            apply_pose("tongue_up")
        elif command == "smile":
            apply_pose("smile")
        elif command == "frown":
            apply_pose("frown")      
        elif command == "wide":
            apply_pose("wide")     
        elif command == "narrow":
            apply_pose("narrow")     
        else:
            print("Unknown command")
    except Exception as e:
        print("Error:", e)





