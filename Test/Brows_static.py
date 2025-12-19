"""
Brows.py - 静止版本
舵机保持在中位，不会自动运动
适合调试和手动控制
"""
from machine import Pin
from time import sleep
from servo import Servo

led = Pin(25, Pin.OUT)

# 创建舵机对象
servo_lo = Servo(pin_id=8)

print("=== Brows Static Mode ===")
print("Servo will stay at 90° (neutral)")
print("Press Ctrl+C to stop and control manually")
print("========================")

# 移动到中位并保持
servo_lo.write(90)
print("Servo at 90° - HOLDING")

# 主循环：只闪烁 LED，不动舵机
while True:
    led.value(not led.value())
    sleep(0.5)
