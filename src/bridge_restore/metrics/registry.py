from .pixel import PixelMetric

def build_metrics(configs, device, corruption):
    result = {}
    for name, params in configs.items():
        params = params or {}
        if name in ("mse", "mae", "psnr"):
            if params: raise ValueError("Pixel metrics take no parameters")
            metric = PixelMetric(name)
        elif name in ("ssim", "ms_ssim"):
            from .structural import StructuralMetric
            metric = StructuralMetric(name, **params)
        elif name == "lpips":
            from .perceptual import LPIPSMetric
            metric = LPIPSMetric(device, **params)
        elif name in ("fid", "kid"):
            from .distribution import DistributionMetric
            metric = DistributionMetric(name, device, **params)
        elif name in ("observation_mse", "observation_mae"):
            from .consistency import ObservationErrorMetric
            if params: raise ValueError("Observation metrics take no parameters")
            metric = ObservationErrorMetric(corruption, name.split("_")[1])
        else: raise ValueError(f"Unknown metric: {name}")
        result[name] = metric
    return result
