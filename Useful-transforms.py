# Editor: Max

# Create Time: 4/2/22 16:33
from PIL import Image
from torchvision import transforms

image_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(image_path)
trans_toTensor = transforms.ToTensor()
img_trans = trans_toTensor(img)


# 标准化使用
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_trans)

