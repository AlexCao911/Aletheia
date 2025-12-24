"""
手动舵机测试脚本
在 Thonny 中运行，可以手动输入角度
"""
from servo import Servo
from machine import Pin

# 创建舵机
servo = Servo(pin_id=8)
led = Pin(25, Pin.OUT)

print("=== 舵机手动测试 ===")
print("舵机连接在 GP8")
print("输入角度 (0-180) 来控制舵机")
print("输入 'q' 退出")
print("==================")

# 先移动到中位
servo.write(90)
print("初始位置: 90°")

while True:
    try:
        # 读取用户输入
        user_input = input("输入角度: ")
        
        if user_input.lower() == 'q':
            print("退出测试")
            servo.write(90)  # 回到中位
            break
        
        # 转换为整数
        angle = int(user_input)
        
        # 检查范围
        if 0 <= angle <= 180:
            servo.write(angle)
            print(f"✓ 移动到 {angle}°")
            led.toggle()  # LED 闪烁表示执行
        else:
            print("✗ 角度必须在 0-180 之间")
    
    except ValueError:
        print("✗ 请输入有效的数字")
    except KeyboardInterrupt:
        print("\n程序中断")
        servo.write(90)
        break
