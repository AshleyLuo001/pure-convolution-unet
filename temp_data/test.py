import numpy as np
import requests
import torch
import io
from PIL import Image
import matplotlib.pyplot as plt

def dice_coef(y_true, y_pred):
    smooth = 1e-5
    y_true_f = y_true.flatten()/255
    y_pred_f = y_pred.flatten()/255

    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    
url="http://localhost:8000/segment"
from io import BytesIO
image = Image.open('C:/Users/Administrator.SY-202304151755/Desktop/docker/project/temp_data/007.png')
shape = image.size

image_buffer = BytesIO()
image.save(image_buffer,format="png")
image_buffer.seek(0)

files ={"file":image_buffer}
response = requests.post(url, files=files)
response_content = response.content # 获取响应内容，即字节流

 # 将返回的二进制数据保存为本地图像文件
image_np = np.frombuffer(response.content, np.uint8)
# 将一维数组形状变更为图像尺寸
image_np = np.reshape(image_np, (shape[1], shape[0]))
# 保存图像为本地文件（文件格式为 PNG）
file_name = os.path.basename(file_path)
save_path = os.path.join(save_directory, os.path.splitext(file_name)[0] + ".png")
cv2.imwrite(save_path, image_np)
print(f"成功处理文件: {file_path}")
# img_stream = io.BytesIo(response_content)
#df =np.frombuffer(response_content,dtype=np.uint8).reshape((shape[1],shape[0]))
# predicted_mask = Image.open(img_stream) # Convert to numpy array
print("df:",df)
print("最大值：",np.max(df))
print("df:",np.nonzero(df))
true_mask=np.array(Image.open('C:/Users/Administrator.SY-202304151755/Desktop/docker/project/temp_data/labelcol/007mask.png'))
print("mask:",true_mask)
print("maskshape:",true_mask.shape)
print("before:",np.nonzero(true_mask))

#true_mask = cv2.resize(true_mask, (224,224))
#true_mask = Image.open('./007mask.png')

#true_mask=true_mask.resize(224,224)
print(true_mask.shape)
#true_mask = true_mask.astype(np.uint8)
true_mask[true_mask > 127 ]=255

print("after:",np.nonzero(true_mask))
# plt.imshow(true_mask)
# plt.show()

dice_score = dice_coef(df,true_mask)
print("Dice Coefficient:",dice_score)