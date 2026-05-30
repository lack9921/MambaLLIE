import lpips
import torch
import numpy as np
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, net='alex'):
    """Calculate LPIPS.

    Args:
        img, img2 (np.ndarray): images in [0, 255], shape (h, w, c) or (b, h, w, c)
        net (str): backbone network (alex or vgg)

    Returns:
        float: LPIPS score
    """
    if img.ndim == 3:
        img = img[None, ...]
        img2 = img2[None, ...]

    # (b, h, w, c) -> (b, c, h, w) and normalize to [-1, 1]
    img_t = torch.from_numpy(img.transpose(0, 3, 1, 2).astype(np.float32)) / 127.5 - 1.0
    img2_t = torch.from_numpy(img2.transpose(0, 3, 1, 2).astype(np.float32)) / 127.5 - 1.0

    if torch.cuda.is_available():
        img_t = img_t.cuda()
        img2_t = img2_t.cuda()

    model = lpips.LPIPS(net=net).eval()
    if torch.cuda.is_available():
        model = model.cuda()

    with torch.no_grad():
        score = model(img_t, img2_t).item()

    return score
