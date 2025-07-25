import cv2
import torch
from PIL import Image
import matplotlib as plt
import numpy as np
import yaml

def ctrim(x):
    x = x.replace("tensor(","")
    x = x.replace(")","")
    x = float(x)
    x = round(x)
    return x
# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


#with open(r'C:\Users\New User\.cache\torch\hub\ultralytics_yolov5_master\data\coco.yaml','r', encoding='utf-8') as f:
#    data = yaml.safe_load(f)
#class_names = data['names']


# Load image
#image = cv2.imread(r'D:\IT\workspace\sai\categories wise images\Mens Apparel\Casual_Jackets\MP000000002328624_1.jpeg')
#image = cv2.imread(r'C:\Users\New User\Downloads\cat.jpeg')
image = cv2.imread(r'C:\Users\aslam\Downloads\dog.jpg')
#image = cv2.imread(r'C:\Users\New User\Downloads\car.jpg')
# Detect objects
img = Image.open(r'C:\Users\aslam\Downloads\dog.jpg')

results = model(image)
res = results.xyxy[0][0]
x = ctrim(str(res[0]))
y = ctrim(str(res[1]))
xdown = ctrim(str(res[2]))
ydown = ctrim(str(res[3]))

print(x,y,xdown,ydown)
coords=(x,y,xdown,ydown)
cropped_img = img.crop(coords)
# Visualize the results
results.render()
print(results)

  
cv2.imshow("img",image) 
cv2.waitKey(0)
cv2.destroyAllWindows()
cropped_img.show()