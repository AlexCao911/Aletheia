"""
舵机范围测试
自动扫描 0-180° 找出舵机的实际工作范围
"""
from servo import Servo
from time import sleep

servo = Servo(pin_id=8)

print("=== 舵机范围测试 ===")
print("舵机将从 0° 扫描到 180°")
print("观察舵机的实际运动范围")
print("==================")

# 先回到中位
servo.write(90)
print("起始位置: 90°")
sleep(2)

# 从 0 到 180 扫描
print("\n开始扫描...")
for angle in range(0, 181, 10):
    servo.write(angle)
    print(f"当前角度: {angle}°")
    sleep(0.5)

print("\n扫描完成！")
print("舵机回到中位 90°")
servo.write(90)
