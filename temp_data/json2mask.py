import os
import json
import numpy as np
from labelme import utils
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
from PIL import Image 


path = 'C:/Users/Administrator.SY-202304151755/Desktop/1/样例数据第二批/json/007.json'
data = json.load(open(path))

imageData = data.get('imageData')
img = utils.img_b64_to_arr(imageData)   # 原始图像


label_name_to_value = {'_background_': 0}
for shape in sorted(data['shapes'], key=lambda x: x['label']):
    label_name = shape['label']
    if label_name in label_name_to_value:
        label_value = label_name_to_value[label_name]
    else:
        label_value = len(label_name_to_value)  
        label_name_to_value[label_name] = label_value  
        
lbl, _ = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

mask = img_as_ubyte(lbl)
mask = np.uint8(mask) * 255   # 掩码
mask_image = Image.fromarray(mask)
mask_image.save("007mask.png")
plt.subplot(1,2,1)
plt.imshow(img)
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(mask, 'gray')
plt.axis('off')


import os


# 指定包含图像的目录
image_directory = '/path/to/your/images'

# 遍历目录中的图像文件
for filename in os.listdir(image_directory):
    if filename.endswith('.png'):
        image_path = os.path.join(image_directory, filename)
        
        # 使用PIL库打开图像
        img = Image.open(image_path)
        
        # 将标签值从 [0, 255] 修改为 [0, 1]
        img = img.point(lambda p: p // 255)
        
        # 保存修改后的图像
        img.save(image_path)

        print(f'Modified: {filename}')
