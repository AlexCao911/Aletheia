from machine import Pin, PWM

class Servo:
    def __init__(self, pin_id):
        self.pwm = PWM(Pin(pin_id))
        self.pwm.freq(50)  # 50Hz for servos
    
    def write(self, angle):
        # Convert angle (0-180) to duty cycle (1000-9000)
        # Typical servo: 1ms (0°) to 2ms (180°)
        duty = int(1000 + (angle / 180) * 8000)
        self.pwm.duty_u16(duty)
    
    def deinit(self):
        self.pwm.deinit()
