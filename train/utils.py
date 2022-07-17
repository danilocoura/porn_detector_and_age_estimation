import torch
import config as c

def get_device():
    if (c.USING_GPU):
        if (c.USING_PARALLEL):
            if (torch.cuda.is_available()):
                if (torch.cuda.device_count() > 1):
                    return "PARALLEL"
                else:
                    return "GPU"
            else:
                return "CPU"
        else:
            if (torch.cuda.is_available()):
                return "GPU"
            else:
                return "CPU"
    else:
        return "CPU"