import cv2
from experiment2 import ResNet34
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import glob
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = ResNet34().to(device)
net.load_state_dict(torch.load('./save/best5.pt', map_location=device))

test_dir = "../ImageData2/test"

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.464, 0.449, 0.386], std=[0.266, 0.259, 0.274])  # 正则化,四舍五入取三位
])

predictions_list=[]

net.eval()
with torch.no_grad():
    for path in glob.glob(test_dir+'/*.jpg'):
        # print(path[14:])
        image = Image.open(path)
        draw = ImageDraw.Draw(image)
        test_image_tensor = transform(image)
        test_image_tensor = test_image_tensor.reshape((1,3,64,64))
        test_image_tensor.to(device)
        out = net(test_image_tensor)
        ret, prediction = torch.max(out, 1)
        predictions_list.append([path[14:], prediction.numpy()[0]])
        print([path[14:], prediction.numpy()[0]])

predictions = np.array(predictions_list)
np.save('predictions.npy', predictions)

with open('test.txt', 'w') as f:
    for path, prediction in predictions:
        print(path+' '+str(prediction))
        f.write(path+' '+str(prediction)+'\n')
