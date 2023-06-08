import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd 

try:
    import _gridencoder as _backend
except ImportError:
    from .backend import _backend

_gridtype_to_id = {
    'hash': 0,
    'tiled': 1,
}

_interp_to_id = {
    'linear': 0,
    'smoothstep': 1,
}

class _grid_encode(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, embeddings, offsets, per_level_scale, base_resolution, calc_grad_inputs=False, gridtype=0, align_corners=False, interpolation=0, max_level=None):
        # inputs: [B, D], float in [0, 1]
        # embeddings: [sO, C], float
        # offsets: [L + 1], int
        # RETURN: [B, F], float
        print("!!! _grid_encode.forward (nmt0")
        print(f"embeddings: {len(embeddings)}")
        if len(embeddings) != 0:
            print(f"embeddings: {embeddings.min()} {embeddings.max()}")

        print("!!! _grid_encode.forward (nmt1")
        print(f"inputs: {len(inputs)}")
        if len(inputs) != 0:
            print(f"inputs: {inputs.min()} {inputs.max()}")
        inputs = inputs.contiguous()
        print("!!! _grid_encode.forward (nmt2")
        print(f"inputs: {len(inputs)}")
        if len(inputs) != 0:
            print(f"inputs: {inputs.min()} {inputs.max()}")

        B, D = inputs.shape # batch size, coord dim
        L = offsets.shape[0] - 1 # level
        C = embeddings.shape[1] # embedding dim for each level
        S = np.log2(per_level_scale) # resolution multiplier at each level, apply log2 for later CUDA exp2f
        H = base_resolution # base resolution
        print("!!! _grid_encode.forward (nmt3")
        print(f"H: {H}")
        print(f"S: {S}")
        print(f"L: {L}")
        print(f"C: {C}")
        print(f"D: {D}")
        print(f"B: {B}")
        print(f"embeddings: {len(embeddings)}")
        if len(embeddings) != 0:
            print(f"embeddings: {embeddings.min()} {embeddings.max()}")
        print(f"max_level: {max_level}")

        max_level = L if max_level is None else max(min(int(math.ceil(max_level * L)), L), 1)
        print("!!! _grid_encode.forward (nmt4")
        print(f"max_level: {max_level}")


        # manually handle autocast (only use half precision embeddings, inputs must be float for enough precision)
        # if C % 2 != 0, force float, since half for atomicAdd is very slow.
        # This might be the cause for NaNs! I think it does not, because then embeddings would be nan at the next debug line already, but they are not.
        if torch.is_autocast_enabled() and C % 2 == 0:
            embeddings = embeddings.to(torch.half)

        print("!!! _grid_encode.forward (nmt5")
        print(f"embeddings: {len(embeddings)}")
        if len(embeddings) != 0:
            print(f"embeddings: {embeddings.min()} {embeddings.max()}")

        # L first, optimize cache for cuda kernel, but needs an extra permute later
        outputs = torch.empty(L, B, C, device=inputs.device, dtype=embeddings.dtype)
        print("!!! _grid_encode.forward (nmt6")
        print(f"outputs: {len(outputs)}")
        if len(outputs) != 0:
            print(f"outputs: {outputs.min()} {outputs.max()}")
        print(f"embeddings: {len(embeddings)}")
        if len(embeddings) != 0:
            print(f"embeddings: {embeddings.min()} {embeddings.max()}")

        # zero init if we only calculate partial levels
        if max_level < L: outputs.zero_()
        print("!!! _grid_encode.forward (nmt7")
        print(f"outputs: {len(outputs)}")
        if len(outputs) != 0:
            print(f"outputs: {outputs.min()} {outputs.max()}")
        print(f"embeddings: {len(embeddings)}")
        if len(embeddings) != 0:
            print(f"embeddings: {embeddings.min()} {embeddings.max()}")

        if calc_grad_inputs:
            dy_dx = torch.empty(B, L * D * C, device=inputs.device, dtype=embeddings.dtype)
            print("!!! _grid_encode.forward (nmt8")
            print(f"dy_dx: {len(dy_dx)}")
            if len(dy_dx) != 0:
                print(f"dy_dx: {dy_dx.min()} {dy_dx.max()}")
            print(f"embeddings: {len(embeddings)}")
            if len(embeddings) != 0:
                print(f"embeddings: {embeddings.min()} {embeddings.max()}")
            if max_level < L: dy_dx.zero_()
            print("!!! _grid_encode.forward (nmt9")
            print(f"dy_dx: {len(dy_dx)}")
            if len(dy_dx) != 0:
                print(f"dy_dx: {dy_dx.min()} {dy_dx.max()}")
            print(f"embeddings: {len(embeddings)}")
            if len(embeddings) != 0:
                print(f"embeddings: {embeddings.min()} {embeddings.max()}")
        else:
            dy_dx = None

        print("!!! Before grid_encode_forward")
        print(f"inputs: {len(inputs)}")
        if len(inputs) != 0:
            print(f"inputs: {inputs.min()} {inputs.max()}")
        print(f"embeddings: {len(embeddings)}")
        if len(embeddings) != 0:
            print(f"embeddings: {embeddings.min()} {embeddings.max()}")
        print(f"offsets: {len(offsets)}")
        if len(offsets) != 0:
            print(f"offsets: {offsets.min()} {offsets.max()}")
        print(f"outputs: {len(outputs)}")
        if len(outputs) != 0:
            print(f"outputs: {outputs.min()} {outputs.max()}")
        print(f"B: {B}")
        print(f"D: {D}")
        print(f"C: {C}")
        print(f"L: {L}")
        print(f"max_level: {max_level}")
        print(f"S: {S}")
        print(f"H: {H}")
        if dy_dx is not None:
            print(f"dy_dx: {len(dy_dx)}")
            if len(dy_dx) != 0:
                print(f"dy_dx: {dy_dx.min()} {dy_dx.max()}")
        else:
            print(f"dy_dx: {dy_dx}")
        print(f"gridtype: {gridtype}")
        print(f"align_corners: {align_corners}")
        print(f"interpolation: {interpolation}")

        _backend.grid_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, max_level, S, H, dy_dx, gridtype, align_corners, interpolation)

        print("!!! After grid_encode_forward")
        print(f"inputs: {len(inputs)}")
        if len(inputs) != 0:
            print(f"inputs: {inputs.min()} {inputs.max()}")
        print(f"embeddings: {len(embeddings)}")
        if len(embeddings) != 0:
            print(f"embeddings: {embeddings.min()} {embeddings.max()}")
        print(f"offsets: {len(offsets)}")
        if len(offsets) != 0:
            print(f"offsets: {offsets.min()} {offsets.max()}")
        print(f"outputs: {len(outputs)}")
        if len(outputs) != 0:
            print(f"outputs: {outputs.min()} {outputs.max()}")
        print(f"B: {B}")
        print(f"D: {D}")
        print(f"C: {C}")
        print(f"L: {L}")
        print(f"max_level: {max_level}")
        print(f"S: {S}")
        print(f"H: {H}")
        if dy_dx is not None:
            print(f"dy_dx: {len(dy_dx)}")
            if len(dy_dx) != 0:
                print(f"dy_dx: {dy_dx.min()} {dy_dx.max()}")
        else:
            print(f"dy_dx: {dy_dx}")
        print(f"gridtype: {gridtype}")
        print(f"align_corners: {align_corners}")
        print(f"interpolation: {interpolation}")


        # permute back to [B, L * C]
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)
        
        print("!!! _grid_encode.forward (nmt10")
        print(f"outputs: {len(outputs)}")
        if len(outputs) != 0:
            print(f"outputs: {outputs.min()} {outputs.max()}")
        print(f"embeddings: {len(embeddings)}")
        if len(embeddings) != 0:
            print(f"embeddings: {embeddings.min()} {embeddings.max()}")

        # I dont think so but maybe this is causing NaNs
        ctx.save_for_backward(inputs, embeddings, offsets, dy_dx)
        print("!!! _grid_encode.forward (nmt11")
        print(f"embeddings: {len(embeddings)}")
        if len(embeddings) != 0:
            print(f"embeddings: {embeddings.min()} {embeddings.max()}")
        ctx.dims = [B, D, C, L, S, H, gridtype, interpolation, max_level]
        print("!!! _grid_encode.forward (nmt12")
        print(f"embeddings: {len(embeddings)}")
        if len(embeddings) != 0:
            print(f"embeddings: {embeddings.min()} {embeddings.max()}")
        ctx.align_corners = align_corners
        print("!!! _grid_encode.forward (nmt13")
        print(f"embeddings: {len(embeddings)}")
        if len(embeddings) != 0:
            print(f"embeddings: {embeddings.min()} {embeddings.max()}")

        return outputs
    
    @staticmethod
    #@once_differentiable
    @custom_bwd
    def backward(ctx, grad):

        # Maybe this is causing NaNs
        inputs, embeddings, offsets, dy_dx = ctx.saved_tensors
        B, D, C, L, S, H, gridtype, interpolation, max_level = ctx.dims
        align_corners = ctx.align_corners

        # grad: [B, L * C] --> [L, B, C]
        grad = grad.view(B, L, C).permute(1, 0, 2).contiguous()

        grad_embeddings = torch.zeros_like(embeddings)

        if dy_dx is not None:
            grad_inputs = torch.zeros_like(inputs, dtype=embeddings.dtype)
        else:
            grad_inputs = None

        # Maybe this is causing NaNs
        _backend.grid_encode_backward(grad, inputs, embeddings, offsets, grad_embeddings, B, D, C, L, max_level, S, H, dy_dx, grad_inputs, gridtype, align_corners, interpolation)

        if dy_dx is not None:
            grad_inputs = grad_inputs.to(inputs.dtype)

        return grad_inputs, grad_embeddings, None, None, None, None, None, None, None, None
        


grid_encode = _grid_encode.apply


class GridEncoder(nn.Module):
    def __init__(self, input_dim=3, num_levels=16, level_dim=2, per_level_scale=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=None, gridtype='hash', align_corners=False, interpolation='linear'):
        super().__init__()

        # the finest resolution desired at the last level, if provided, overridee per_level_scale
        if desired_resolution is not None:
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))

        self.input_dim = input_dim # coord dims, 2 or 3
        self.num_levels = num_levels # num levels, each level multiply resolution by 2
        self.level_dim = level_dim # encode channels per level
        self.per_level_scale = per_level_scale # multiply resolution by this scale at each level.
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim
        self.gridtype = gridtype
        self.gridtype_id = _gridtype_to_id[gridtype] # "tiled" or "hash"
        self.interpolation = interpolation
        self.interp_id = _interp_to_id[interpolation] # "linear" or "smoothstep"
        self.align_corners = align_corners

        # allocate parameters
        offsets = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size
        for i in range(num_levels):
            resolution = int(np.ceil(base_resolution * per_level_scale ** i))
            params_in_level = min(self.max_params, (resolution) ** input_dim) # limit max number
            params_in_level = int(np.ceil(params_in_level / 8) * 8) # make divisible
            offsets.append(offset)
            offset += params_in_level
        offsets.append(offset)
        offsets = torch.from_numpy(np.array(offsets, dtype=np.int32))
        self.register_buffer('offsets', offsets)
        
        self.n_params = offsets[-1] * level_dim

        # parameters
        self.embeddings = nn.Parameter(torch.empty(offset, level_dim))
        print("!!!GridEncoder.__init__ (hjk1")
        print(f"embeddings: {len(self.embeddings)}")
        if len(self.embeddings) != 0:
            print(f"embeddings: {self.embeddings.min()} {self.embeddings.max()}")

        self.reset_parameters()
        print("!!!GridEncoder.__init__ (hjk1")
        print(f"embeddings: {len(self.embeddings)}")
        if len(self.embeddings) != 0:
            print(f"embeddings: {self.embeddings.min()} {self.embeddings.max()}")
    
    def reset_parameters(self):
        std = 1e-4
        self.embeddings.data.uniform_(-std, std)

    def __repr__(self):
        return f"GridEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} resolution={self.base_resolution} -> {int(round(self.base_resolution * self.per_level_scale ** (self.num_levels - 1)))} per_level_scale={self.per_level_scale:.4f} params={tuple(self.embeddings.shape)} gridtype={self.gridtype} align_corners={self.align_corners} interpolation={self.interpolation}"
    
    def forward(self, inputs, bound=1, max_level=None):
        # inputs: [..., input_dim], normalized real world positions in [-bound, bound]
        # max_level: only calculate first max_level levels (None will use all levels)
        # return: [..., num_levels * level_dim]

        print("!!!GridEncoder.forward (rtu1")
        print(f"embeddings: {len(self.embeddings)}")
        if len(self.embeddings) != 0:
            print(f"embeddings: {self.embeddings.min()} {self.embeddings.max()}")
        print(f"inputs: {len(inputs)}")
        if len(inputs) != 0:
            print(f"inputs: {inputs.min()} {inputs.max()}")
        inputs = (inputs + bound) / (2 * bound) # map to [0, 1]
        print("!!!GridEncoder.forward (rtu2")
        print(f"embeddings: {len(self.embeddings)}")
        if len(self.embeddings) != 0:
            print(f"embeddings: {self.embeddings.min()} {self.embeddings.max()}")
        print(f"inputs: {len(inputs)}")
        if len(inputs) != 0:
            print(f"inputs: {inputs.min()} {inputs.max()}")
        
        #print('inputs', inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item())

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)
        print("!!!GridEncoder.forward (rtu3")
        print(f"embeddings: {len(self.embeddings)}")
        if len(self.embeddings) != 0:
            print(f"embeddings: {self.embeddings.min()} {self.embeddings.max()}")
        print(f"inputs: {len(inputs)}")
        if len(inputs) != 0:
            print(f"inputs: {inputs.min()} {inputs.max()}")

        outputs = grid_encode(inputs, self.embeddings, self.offsets, self.per_level_scale, self.base_resolution, inputs.requires_grad, self.gridtype_id, self.align_corners, self.interp_id, max_level)
        outputs = outputs.view(prefix_shape + [self.output_dim])
        print("!!!GridEncoder.forward (rtu4")
        print(f"embeddings: {len(self.embeddings)}")
        if len(self.embeddings) != 0:
            print(f"embeddings: {self.embeddings.min()} {self.embeddings.max()}")
        print(f"outputs: {len(outputs)}")
        if len(outputs) != 0:
            print(f"outputs: {outputs.min()} {outputs.max()}")

        #print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())

        return outputs

    # always run in float precision!
    @torch.cuda.amp.autocast(enabled=False)
    def grad_total_variation(self, weight=1e-7, inputs=None, bound=1, B=1000000):
        # inputs: [..., input_dim], float in [-b, b], location to calculate TV loss.
        print("!!!GridEncoder.grad_total_variation (lki1")
        print(f"embeddings: {len(self.embeddings)}")
        if len(self.embeddings) != 0:
            print(f"embeddings: {self.embeddings.min()} {self.embeddings.max()}")
        D = self.input_dim
        C = self.embeddings.shape[1] # embedding dim for each level
        L = self.offsets.shape[0] - 1 # level
        S = np.log2(self.per_level_scale) # resolution multiplier at each level, apply log2 for later CUDA exp2f
        H = self.base_resolution # base resolution
        print("!!!GridEncoder.grad_total_variation (lki2")
        print(f"embeddings: {len(self.embeddings)}")
        if len(self.embeddings) != 0:
            print(f"embeddings: {self.embeddings.min()} {self.embeddings.max()}")

        if inputs is None:
            # randomized in [0, 1]
            inputs = torch.rand(B, self.input_dim, device=self.embeddings.device)
        else:
            inputs = (inputs + bound) / (2 * bound) # map to [0, 1]
            inputs = inputs.view(-1, self.input_dim)
            B = inputs.shape[0]

        print("!!!GridEncoder.grad_total_variation (lki3")
        print(f"embeddings: {len(self.embeddings)}")
        if len(self.embeddings) != 0:
            print(f"embeddings: {self.embeddings.min()} {self.embeddings.max()}")

        if self.embeddings.grad is None:
            raise ValueError('grad is None, should be called after loss.backward() and before optimizer.step()!')
        
        print("!!!GridEncoder.grad_total_variation (lki4")
        print(f"embeddings: {len(self.embeddings)}")
        if len(self.embeddings) != 0:
            print(f"embeddings: {self.embeddings.min()} {self.embeddings.max()}")

        _backend.grad_total_variation(inputs, self.embeddings, self.embeddings.grad, self.offsets, weight, B, D, C, L, S, H, self.gridtype_id, self.align_corners)

        print("!!!GridEncoder.grad_total_variation (lki5")
        print(f"embeddings: {len(self.embeddings)}")
        if len(self.embeddings) != 0:
            print(f"embeddings: {self.embeddings.min()} {self.embeddings.max()}")
    
    @torch.cuda.amp.autocast(enabled=False)
    def grad_weight_decay(self, weight=0.1):
        # level-wise meaned weight decay (ref: zip-nerf)
        
        print("!!!GridEncoder.grad_weight_decay (tip1")
        print(f"embeddings: {len(self.embeddings)}")
        if len(self.embeddings) != 0:
            print(f"embeddings: {self.embeddings.min()} {self.embeddings.max()}")

        B = self.embeddings.shape[0] # size of embedding
        C = self.embeddings.shape[1] # embedding dim for each level
        L = self.offsets.shape[0] - 1 # level
        print("!!!GridEncoder.grad_weight_decay (tip2")
        print(f"embeddings: {len(self.embeddings)}")
        if len(self.embeddings) != 0:
            print(f"embeddings: {self.embeddings.min()} {self.embeddings.max()}")
        
        if self.embeddings.grad is None:
            raise ValueError('grad is None, should be called after loss.backward() and before optimizer.step()!')
        
        print("!!!GridEncoder.grad_weight_decay (tip3")
        print(f"embeddings: {len(self.embeddings)}")
        if len(self.embeddings) != 0:
            print(f"embeddings: {self.embeddings.min()} {self.embeddings.max()}")

        _backend.grad_weight_decay(self.embeddings, self.embeddings.grad, self.offsets, weight, B, C, L)

        print("!!!GridEncoder.grad_weight_decay (tip4")
        print(f"embeddings: {len(self.embeddings)}")
        if len(self.embeddings) != 0:
            print(f"embeddings: {self.embeddings.min()} {self.embeddings.max()}")