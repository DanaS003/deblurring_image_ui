# utils.py

import torch
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import torch.nn.functional as F
import numpy as np
from PIL import Image

# These functions are defined *here*, not imported from elsewhere in utils
inference_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def denormalize_img(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(tensor.device)
    return (tensor * std + mean).clamp(0, 1)

def calculate_metrics(deblurred_tensor, ground_truth_tensor):
    # Ensure sizes are compatible for metric calculation
    if deblurred_tensor.shape[1:] != ground_truth_tensor.shape[1:]:
        ground_truth_tensor = F.interpolate(
            ground_truth_tensor.unsqueeze(0),
            size=deblurred_tensor.shape[1:],
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

    deblur_np = deblurred_tensor.permute(1, 2, 0).cpu().numpy()
    gt_np = ground_truth_tensor.permute(1, 2, 0).cpu().numpy()

    psnr = psnr_metric(gt_np, deblur_np, data_range=1)
    ssim = ssim_metric(gt_np, deblur_np, data_range=1, channel_axis=-1)
    return psnr, ssim