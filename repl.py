import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np

width = 512
height = 512
thres_1 = 200
thres_2 = 400

image = torch.rand(1, 1600, 1208, 3)
print(image.shape)


print(image.shape)

# to_pil = T.ToPILImage()
# img = to_pil(image)
# img = img.crop((0, 0, width, height))
# print(img.size)
# img.show()


# to_tensor = T.Compose([T.ToTensor()])

# tensor = to_tensor(img)
# tensor = tensor.unsqueeze_(0).permute(0, 3, 2, 1)
# print(tensor.shape)


# Convert the tensor to a numpy array
image = image.squeeze_().permute(2, 1, 0)
numpy_image = image.numpy()

print(numpy_image.shape)
# opencv_image = (numpy_image * 255).astype(np.uint8)
gray_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2GRAY)
# canny_img = cv2.Canny(gray_image, thres_1, thres_2)
# canny_img_normalized = canny_img / 255.0
# canny_tensor = torch.from_numpy(canny_img_normalized)
# canny_tensor = canny_tensor.unsqueeze(0) 
# print(canny_tensor.shape)