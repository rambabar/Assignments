from typing import Any, Dict, Optional, Union

import torch
import torchvision.transforms as transforms

from kornia.core import Module, Tensor
from kornia.geometry import resize
from kornia.utils.helpers import map_location_to_cpu

from kornia.feature.loftr.loftr import LoFTR, default_cfg

# default_cfg = {
#     'backbone_type': 'ResNetFPN',
#     'resolution': (8, 2),
#     'fine_window_size': 5,
#     'fine_concat_coarse_feat': True,
#     'resnetfpn': {'initial_dim': 128, 'block_dims': [128, 196, 256]},
#     'coarse': {
#         'd_model': 256,
#         'd_ffn': 256,
#         'nhead': 8,
#         'layer_names': ['self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross'],
#         'attention': 'linear',
#         'temp_bug_fix': False,
#     },
#     'match_coarse': {
#         'thr': 0.2,
#         'border_rm': 2,
#         'match_type': 'dual_softmax',
#         'dsmax_temperature': 0.1,
#         'skh_iters': 3,
#         'skh_init_bin_score': 1.0,
#         'skh_prefilter': True,
#         'train_coarse_percent': 0.4,
#         'train_pad_num_gt_min': 200,
#     },
#     'fine': {'d_model': 128, 'd_ffn': 128, 'nhead': 8, 'layer_names': ['self', 'cross'], 'attention': 'linear'},
# }

class LoFTRInference:
    def __init__(self, pretrained: Optional[str] = 'outdoor', config: Dict[str, Any] = default_cfg):
        self.loftr = LoFTR(pretrained, config)
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert image to Tensor
            # Add any other preprocessing transforms you may need
        ])

    def preprocess_image(self, image):
        # Preprocess the input image
        # You may need to resize, normalize, or apply other transformations
        preprocessed_image = self.transform(image)
        return preprocessed_image

    def postprocess_result(self, result):
        # Postprocess the model output if needed
        # Extract the required information from the result dictionary
        # You can also visualize the matching or perform any other custom processing
        return result

    def infer(self, image0, image1):
        # Preprocess the input images
        preprocessed_image0 = self.preprocess_image(image0)
        preprocessed_image1 = self.preprocess_image(image1)

        # Prepare the input data dictionary
        input_data = {"image0": preprocessed_image0.unsqueeze(0), "image1": preprocessed_image1.unsqueeze(0)}

        # Perform inference using the LoFTR model
        result = self.loftr(input_data)

        # Postprocess the result
        output = self.postprocess_result(result)

        return output

import cv2,  time
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# import matplotlib.pyplot as plt

# Create an instance of the LoFTRInference class
loftr = LoFTRInference()

def matching_operation(image1, image2):
    
    # Resize the images to have the same dimensions
    image1 = cv2.resize(image1, (300, 200))  # Adjust the dimensions as needed
    image2 = cv2.resize(image2, (300, 200))  # Adjust the dimensions as needed

    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Convert the images to tensors
    image1_tensor = torch.from_numpy(image1_gray).unsqueeze(0).float() / 255.0
    image2_tensor = torch.from_numpy(image2_gray).unsqueeze(0).float() / 255.0

    # Remove the extra dimensions from tensors
    image1_tensor = image1_tensor.squeeze(0)
    image2_tensor = image2_tensor.squeeze(0)

    # Convert tensors to PIL Images
    image1_pil = transforms.ToPILImage()(image1_tensor)
    image2_pil = transforms.ToPILImage()(image2_tensor)
    
    t1 = time.time()
    # Perform inference
    output = loftr.infer(image1_pil, image2_pil)
    print("time for actual inference:", time.time() - t1)

    # Access the output values
    keypoints1 = output['keypoints0']  # Matching keypoints from image1
    keypoints2 = output['keypoints1']  # Matching keypoints from image2
    confidence = output['confidence']  # Confidence scores

    # Draw keypoints on the images
    for kp in keypoints1:
        x, y = kp[0], kp[1]
        cv2.circle(image1, (int(x), int(y)), 3, (0, 255, 0), -1)

    for kp in keypoints2:
        x, y = kp[0], kp[1]
        cv2.circle(image2, (int(x), int(y)), 3, (0, 255, 0), -1)

    # Combine the images side by side
    combined_image = np.concatenate((image1, image2), axis=1)

    # Convert the combined image to RGB format
    combined_image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)

    return combined_image_rgb
# image1_path = os.path.join(os.getcwd(),'image3.jpg')
# image2_path = os.path.join(os.getcwd(),'image4.jpg')
# output_image_path = os.path.join(os.getcwd(),'output_image.jpg')

# t1 = time.time()
# # Load the input images
# image1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
# image2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)

# output_image = matching_operation(image1, image2)
# print("total time taken:", time.time()-t1)

# cv2.imshow("Output Image", output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# from google.colab.patches import cv2_imshow
# cv2_imshow(cv2.imread(output_image_path))

# import matplotlib.pyplot as plt
# plt.imshow(output_image)
# plt.axis('off')
# plt.show()