# Editor: Max

# Create Time: 4/2/22 15:27
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import cv2

image_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(image_path)

writer = SummaryWriter("logs")

# transforms如何使用
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

# 利用opencv来导入图片
# cv_img = cv2.imread(image_path)
# print(type(cv_img))

writer.add_image("img", tensor_img)

writer.close()
