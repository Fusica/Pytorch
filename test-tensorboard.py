# Editor: Max

# Create Time: 4/2/22 10:53

from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "dataset/train/ants_image/0013035.jpg"
img_PIL = Image.open(image_path)
image_array = np.array(img_PIL)

writer.add_image("test", image_array, 1, dataformats='HWC')

# writer.add_image()
# y = x
for i in range(100):
    writer.add_scalar("y=x^2", i*i, i)

writer.close()
