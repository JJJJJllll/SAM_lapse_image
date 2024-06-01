import cv2
import matplotlib.pyplot as plt
import numpy as np

# 读取图像
img = cv2.imread('C0004.00_00_32_16.Still001.jpg')  # 替换为你的图像文件路径
if img is None:
    print("无法读取图像")
    exit()

# 将BGR图像转换为RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 使用matplotlib显示图像
fig, ax = plt.subplots()
ax.imshow(img_rgb)

# 准备一个点
x, y = 0, 0

# 定义一个回调函数，当用户点击图像时将被调用
def onclick(event):
    if event.dblclick:
        global x, y
        x, y = int(event.xdata), int(event.ydata)
        print(f'像素位置: x={x}, y={y}')
        ax.plot(x, y, 'ro')  # 在点击位置画一个红点
        plt.draw()

# 将回调函数与图像窗口关联
fig.canvas.mpl_connect('button_press_event', onclick)

# 显示图像窗口
plt.show()

print(x, y)