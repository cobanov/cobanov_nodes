import torchvision.transforms as T
import numpy as np
import torch
import cv2

MAX_RESOLUTION = 8192


class CobanovNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": (
                    "INT",
                    {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "x": (
                    "INT",
                    {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8},
                ),
                "y": (
                    "INT",
                    {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop"

    CATEGORY = "latent/transform"

    def crop(self, image, width, height, x, y):
        image = image.squeeze_().permute(2, 1, 0)

        transform = T.ToPILImage()
        img = transform(image)
        img = img.crop((0, 0, width, height))

        to_tensor = T.Compose([T.ToTensor()])

        tensor = to_tensor(img)
        tensor = tensor.unsqueeze_(0).permute(0, 3, 2, 1)
        print(tensor.shape)

        return (tensor,)


class CobanovNode_Canny:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "thres_1": (
                    "INT",
                    {"default": 100, "min": 0, "max": 500, "step": 8},
                ),
                "thres_2": (
                    "INT",
                    {"default": 200, "min": 0, "max": 500, "step": 8},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "canny"

    CATEGORY = "latent/transform"

    def canny(self, image, thres_1, thres_2):
        image_squeeze = image.squeeze_()
        numpy_image = image_squeeze.numpy()
        opencv_image = (numpy_image * 255).astype(np.uint8)
        gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        canny_img = cv2.Canny(gray_image, thres_1, thres_2)
        canny_tensor = torch.from_numpy(canny_img).unsqueeze(0)

        return (canny_tensor,)


NODE_CLASS_MAPPINGS = {
    "CobanovNode": CobanovNode,
    "CobanovCanny": CobanovNode_Canny,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CobanovNode": "Cobanov Custom Node",
}
