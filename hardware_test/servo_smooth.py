"""
改进的 Servo 类
添加平滑移动功能，减少电流冲击
"""
from machine import Pin, PWM
from time import sleep_ms

class Servo:
    def __init__(self, pin_id):
        self.pwm = PWM(Pin(pin_id))
        self.pwm.freq(50)
        self.current_angle = 90  # 记录当前角度
        self.write(90)  # 初始化到中位
        print(f"Servo initialized on pin {pin_id}")
    
    def write(self, angle):
        """
        直接移动到目标角度
        """
        angle = max(0, min(180, angle))
        duty = int(1640 + (angle / 180) * (8190 - 1640))
        self.pwm.duty_u16(duty)
        self.current_angle = angle
    
    def write_smooth(self, target_angle, step=2, delay=20):
        """
        平滑移动到目标角度
        
        参数：
        - target_angle: 目标角度 (0-180)
        - step: 每步移动的角度 (默认 2°)
        - delay: 每步之间的延迟 (毫秒，默认 20ms)
        """
        target_angle = max(0, min(180, target_angle))
        
        # 计算移动方向
        if target_angle > self.current_angle:
            direction = 1
        elif target_angle < self.current_angle:
            direction = -1
        else:
            return  # 已经在目标位置
        
        # 逐步移动
        while True:
            # 计算下一个角度
            next_angle = self.current_angle + (direction * step)
            
            # 检查是否超过目标
            if direction == 1 and next_angle >= target_angle:
                self.write(target_angle)
                break
            elif direction == -1 and next_angle <= target_angle:
                self.write(target_angle)
                break
            else:
                self.write(next_angle)
                sleep_ms(delay)
        
        print(f"Moved to {target_angle}°")
    
    def stop(self):
        """停止 PWM 信号"""
        self.pwm.duty_u16(0)
    
    def deinit(self):
        """释放 PWM"""
        self.pwm.deinit()
