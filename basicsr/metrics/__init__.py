from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .psnr_ssim import calculate_psnr, calculate_ssim
from .calculate_lpips import calculate_lpips

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_lpips']


def calculate_metric(data, opt):
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
