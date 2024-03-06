
from math import inf
import torch
from torch import tensor, device
import torch.fx as fx
import torch._dynamo
from torch._dynamo.testing import rand_strided
from torch._dynamo.debug_utils import run_fwd_maybe_bwd

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config









from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()



    def forward(self, h_relu1_2):
        return (h_relu1_2,)


mod = Repro()

def load_args(reader):
    buf0 = reader.storage('4f938c4884d65723101a0cdbfd741e5bb26ad651', 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (64, 64, 64, 64), (262144, 1, 4096, 64), dtype=torch.bfloat16, requires_grad=True)  # h_relu1_2
load_args._version = 0

if __name__ == '__main__':
    from torch._dynamo.repro.after_dynamo import run_repro
    run_repro(mod, load_args, accuracy=False, command='run',
        save_dir='/home/drhead/PerceptualSimilarity/checkpoints', autocast=True, backend='eager')
