import face_alignment as fa

import torch
import cv2

MAX_RESOLUTION = 8192


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def draw_landmarks(img, landmarks):
    lm_image = img.copy()

    for x, y in landmarks[0]:
        coordinates = (int(x), int(y))
        lm_image = cv2.circle(lm_image, coordinates, 2, (50, 255, 50), 2)
    return lm_image


class CobanovFace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_landmark"

    CATEGORY = "latent/transform"
    RETURN_NAMES = ("IMAGES", "MASKS")
    RETURN_TYPES = ("IMAGE", "MASK")

    def get_landmark(self, image):
        fa = fa.FaceAlignment(
            fa.LandmarksType.TWO_D, flip_input=False, device=get_device()
        )
        preds = fa.get_landmarks(image)
        face_landmark = preds[0]

        lm_image = draw_landmarks(image, face_landmark)

        return (lm_image, face_landmark)


NODE_CLASS_MAPPINGS = {
    "CobanovNode": CobanovFace,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CobanovNode": "Cobanov Custom Node",
}
