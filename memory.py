import torch
from dataclasses import dataclass

@dataclass
class GibMemoryUsage:
    allocated: float
    reserved: float
    peak: float

def measure_memory() -> GibMemoryUsage:
    torch.cuda.synchronize()
    scale = 2**30

    return GibMemoryUsage(
        allocated=round(torch.cuda.memory_allocated() / scale, 3),
        reserved=round(torch.cuda.memory_reserved() / scale, 3),
        peak=round(torch.cuda.max_memory_allocated() / scale, 3)
    )
