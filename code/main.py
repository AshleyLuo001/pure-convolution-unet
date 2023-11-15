import io
import uvicorn
import numpy as np
from PIL import Image
import sys
sys.path.append('./')

import torch
import torchvision.transforms as transforms
#from torchvision.models.segmentation import deeplabv3_resnet101
from fastapi import FastAPI, UploadFile, File,Response,HTTPException
import torch.optim
from temp_data.Load_Dataset import ValGenerator, ImageToImage2D,correct_dims
from torch.utils.data import DataLoader
import warnings
import pickle
warnings.filterwarnings("ignore")
import temp_data.Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import os
from scipy.ndimage import zoom
import shutil
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from temp_data.ACC_UNet import ACC_UNet
from PIL import Image
import json
from temp_data.utils import *
import cv2


def show_image_with_dice(predict_save, labs, save_path):

    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))


    return dice_pred, iou_pred

device = torch.device('cpu')
app = FastAPI()
output_size=(224,224)
# 创建模型实例
model_path = './model/best_model-ACC_UNet.pth.tar'
temp_data ='./temp_data/'
checkpoint = torch.load(model_path, map_location='cpu')
config_vit = config.get_CTranS_config()   
model = ACC_UNet(n_channels=config.n_channels,n_classes=config.n_labels,n_filts=config.n_filts)
model.to(device)

# 使用cpu加载权重
model_path = './model/best_model-ACC_UNet.pth.tar'
model.load_state_dict(checkpoint['state_dict'])
print('Model loaded !')



model.eval()

# 定义路由和处理函数
@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):

    #如果文件读取错误，则返回错误
    try:
        image = await file.read()
        img_file = io.BytesIO(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail="文件读取错误") from e

    # 文件读取成功，按照代码逻辑继续处理
    try:
    # 
        img = Image.open(img_file)
        save_directory = './temp_data/'
        save_directory_1='./temp_data/img'
        file_path = os.path.join(save_directory_1, file.filename)
        img.save(file_path)
        
        tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
        test_dataset = ImageToImage2D(save_directory, tf_test,image_size=config.img_size)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            
            for i, (sampled_batch, names) in enumerate(test_loader, 1):
                test_data=sampled_batch['image']
                arr = test_data.numpy()
                arr = arr.astype(np.float32)

                height, width = config.img_size, config.img_size

                input_img = torch.from_numpy(arr)


                output = model(input_img)
                #通過閾值將模型的輸出二值化 
                pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
                #將二值化的結果轉為numpy數組
                predict_save = pred_class[0].cpu().data.numpy()

                input_img.to('cpu')
                input_img = input_img[0].transpose(0,-1).cpu().detach().numpy()
                output = output[0,0,:,:].cpu().detach().numpy()
                output=(output>=0.5)*1.0 
                output = output.astype(np.uint8)*255
                print('形狀:',output)
                print("最大值：",np.max(output))
                plt.imshow(output)
                plt.show()
                # 将模型预测的结节分割图像转换为字节型数据并返回
                output_bytes = output.tobytes()
                response = Response(content=output_bytes, media_type="application/octet-stream")
                return response

    #如果模型处理后的结果未能正确返回，则以字节流形式返回全是255的二值图
    except:
        img = Image.open(img_file)
        shape = img.size
        empty_image = np.zeros((shape[1], shape[0]), dtype=np.uint8)+255
        empty_bytes = empty_image.tobytes()
        response = Response(content=empty_bytes, media_type="application/octet-stream")
        return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)