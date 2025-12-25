"""
Brows.py - 简单调试版本
只测试一个舵机
"""
from machine import Pin
from time import sleep
from servo import Servo

led = Pin(25, Pin.OUT)

# 创建舵机对象
servo_lo = Servo(pin_id=8)

print("=== Brows Single Servo Test ===")
print("Testing servo on GP8")
print("Moving through test sequence...")

# 测试序列
test_angles = [90, 80, 100, 120, 80, 90]

while True:
    led.value(not led.value())
    
    for angle in test_angles:
        print(f"Moving to {angle}°")
        servo_lo.write(angle)
        sleep(1)  # 停留1秒
    
    print("Test sequence complete. Repeating...")
    sleep(2)
