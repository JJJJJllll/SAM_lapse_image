import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


'''source/target library'''
source_folder = '/home/jsl/图片/前10张调亮'
backup_folder = '/home/jsl/图片/SAM输出'
# 获取源文件夹中的所有文件
files = os.listdir(source_folder) 
# 确保备份文件夹存在
if not os.path.exists(backup_folder):
    os.makedirs(backup_folder)
 

'''load the SAM model and predictor'''
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

device = "cpu"

sam = sam_model_registry["default"](checkpoint="/home/jsl/下载/sam_vit_h_4b8939.pth")
sam.to(device=device)

predictor = SamPredictor(sam)

'''iterate images'''
for file in files:
    name_without_extension, extension = os.path.splitext(file)
    if extension.lower() in ['.png', '.jpg', '.jpeg']:
        # read image
        image_path = os.path.join(source_folder, file)
        image = cv2.imread(image_path)

        '''produce an image embedding'''
        predictor.set_image(image)  

        '''choose a point on image'''
        x, y = 0, 0
        fig, ax = plt.subplots()
        ax.imshow(image)

        def onclick(event):
            if event.dblclick: # 双击
                global x, y
                x, y = int(event.xdata), int(event.ydata)
                print(f'像素位置: x={x}, y={y}')
                ax.plot(x, y, 'ro')  # 在点击位置画一个红点
                plt.draw()

        # 将回调函数与图像窗口关联
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()  

        '''choose a point as input'''
        input_point = np.array([[x, y]])
        input_label = np.array([1])

        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_points(input_point, input_label, plt.gca())
        plt.axis('on')
        plt.show()  

        '''predict'''
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        print(masks.shape)  # (number_of_masks) x H x W)

        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            show_mask(mask, plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            
            # 将布尔mask转换为数值类型，True转换为1，False转换为0
            mask = mask.astype(np.uint8) * 255

            # # 使用mask提取图像
            # extracted_image = cv2.bitwise_and(image, image, mask=mask)

            # # # 保存、显示提取的图像
            # plt.figure(figsize=(10,10))
            # plt.imshow(extracted_image)
            # plt.show()
            # cv2.imwrite(f'{backup_folder}/{name_without_extension}_mask{i}.png', extracted_image)

            # 将掩码转换为4通道（添加alpha通道）
            rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            rgba_image[:, :, 3] = mask  # 将掩码作为alpha通道

            # 保存或显示提取的图像
            plt.figure(figsize=(10,10))
            plt.imshow(rgba_image)
            plt.show()
            cv2.imwrite(f'{backup_folder}/{name_without_extension}_mask{i}.png', rgba_image)

        



  