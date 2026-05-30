import lpips
import torch
import numpy as np
from basicsr.utils.registry import METRIC_REGISTRY

# 缓存 LPIPS 模型，只加载一次
_lpips_model = None

def _get_lpips_model(net='alex'):
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net=net).eval()
        if torch.cuda.is_available():
            _lpips_model = _lpips_model.cuda()
    return _lpips_model


@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, net='alex'):
    if img.ndim == 3:
        img = img[None, ...]
        img2 = img2[None, ...]

    img_t = torch.from_numpy(img.transpose(0, 3, 1, 2).astype(np.float32)) / 127.5 - 1.0
    img2_t = torch.from_numpy(img2.transpose(0, 3, 1, 2).astype(np.float32)) / 127.5 - 1.0

    if torch.cuda.is_available():
        img_t = img_t.cuda()
        img2_t = img2_t.cuda()

    model = _get_lpips_model(net)

    with torch.no_grad():
        score = model(img_t, img2_t).item()

    return score
