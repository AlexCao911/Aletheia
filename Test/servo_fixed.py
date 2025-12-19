"""
改进的 Servo 类
支持更精确的 PWM 控制
"""
from machine import Pin, PWM

class Servo:
    def __init__(self, pin_id):
        self.pwm = PWM(Pin(pin_id))
        self.pwm.freq(50)  # 50Hz for servos
        print(f"Servo initialized on pin {pin_id}")
    
    def write(self, angle):
        """
        控制舵机角度
        angle: 0-180
        """
        # 限制角度范围
        angle = max(0, min(180, angle))
        
        # 标准舵机：1ms (0°) 到 2ms (180°)
        # 50Hz = 20ms 周期
        # duty_u16 范围：0-65535
        # 1ms = 3277, 2ms = 6553
        min_duty = 1640  # 约 0.5ms（更安全的最小值）
        max_duty = 8190  # 约 2.5ms（更安全的最大值）
        
        duty = int(min_duty + (angle / 180) * (max_duty - min_duty))
        self.pwm.duty_u16(duty)
        print(f"Angle: {angle}°, Duty: {duty}")
    
    def write_us(self, microseconds):
        """
        直接设置脉宽（微秒）
        microseconds: 500-2500
        """
        # 50Hz = 20000us 周期
        # duty_u16 = (microseconds / 20000) * 65535
        duty = int((microseconds / 20000) * 65535)
        self.pwm.duty_u16(duty)
        print(f"Pulse width: {microseconds}us, Duty: {duty}")
    
    def stop(self):
        """
        停止 PWM 信号
        """
        self.pwm.duty_u16(0)
        print("PWM stopped")
    
    def deinit(self):
        """
        释放 PWM
        """
        self.pwm.deinit()
        print("Servo deinitialized")
