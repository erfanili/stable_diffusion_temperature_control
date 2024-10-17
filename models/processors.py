import torch
import torch.nn.functional as F 
from typing import Optional
from diffusers.utils import deprecate, logging
from diffusers.models.attention_processor import Attention
import math
import numpy as  np 
import matplotlib.pyplot as plt





import torch
import torch.nn.functional as F 
from typing import Optional
from diffusers.utils import deprecate, logging
from diffusers.models.attention_processor import Attention
import math
import torch.nn as nn


  
    
class AttnProcessor3:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self,positive_prompt:bool = True,idx1=[],idx2=[], eos_idx=[]):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.idx1 = idx1
        self.idx2 = idx2
        self.eos_idx = eos_idx
        self.positive_prompt = positive_prompt


    def correct_it(self,attention_probs:torch.Tensor, idx1, idx2, eos_idx, timestep:int, alpha1:float=2, alpha2:float=3)-> torch.Tensor:

        def gaussian_kernel(size: int, sigma: float):
            """Generates a 2D Gaussian kernel."""
            x = torch.arange(-size // 2 + 1., size // 2 + 1.)
            y = torch.arange(-size // 2 + 1., size // 2 + 1.)
            x_grid, y_grid = torch.meshgrid(x, y)
            kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
            kernel = kernel / torch.sum(kernel)
            return kernel

        def normalize(tensor,dim=1):
            mu = tensor.mean(dim, keepdim=True)
            std = tensor.std(dim, keepdim=True)
            output = (tensor - mu)/(std+1e-5)
            return output,mu,std 
        
        def denormalize(tensor,mu,std):
            return (tensor*(std+1e-5))+mu
        
        def get_outlier_idx(tensor):
            _, idx = torch.sort(tensor[0], descending=True)
            # tensor[0]
            return

        attention_probs_positive = attention_probs[16:]
        kernel = 1
        attention_probs_positive_reshape = attention_probs_positive.reshape(16, 32, 32, -1)
        tensor_permuted = attention_probs_positive_reshape.permute(0, 3, 1, 2) # 16, 120, 32, 32
        pooled = F.avg_pool2d(tensor_permuted, kernel_size=(kernel, kernel))  # 16, 120, 16, 16
        attention_probs_positive_pooled = pooled.permute(0, 2, 3, 1) # 16, 16, 16, 120
        attention_probs_positive_pooled = attention_probs_positive_pooled.reshape(16, int(1024/(kernel*kernel)), -1)
        
        attn_scr1 = attention_probs_positive_pooled[:,:,idx1]
        attn_scr2 = attention_probs_positive_pooled[:,:,idx2]
    
        condition1 = attn_scr1 > attn_scr2*1.3
        condition2 = attn_scr1*1.3 <= attn_scr2

        if (torch.sum(condition1) < 2400 or torch.sum(condition2) < 2400) and timestep > 800:
            
            noise = torch.rand(1, 1, int(32/kernel), int(32/kernel))  # Shape: (batch_size, channels, height, width)
            gaussian_kernel = torch.tensor([[[[1, 2, 1], [1, 1, 2], [2, 1, 1]]]], dtype=torch.float32)
            gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
            smoothed_noise = F.conv2d(noise, gaussian_kernel, padding=1)
            mask = smoothed_noise > 0.5 
            binary_mask = mask.squeeze(0).squeeze(0)
            condition1 = binary_mask.flatten()
            condition2 = ~binary_mask.flatten()
            condition1 = condition1.unsqueeze(0).repeat(16, 1)
            condition2 = condition2.unsqueeze(0).repeat(16, 1)
            
        attn_sum_except_eos = torch.sum(attention_probs_positive_pooled[:,:,:eos_idx], axis=-1)

        attention_probs_positive_pooled[:,:,idx1][condition1] = attn_sum_except_eos[condition1]
        for i in range(eos_idx):
            if i != idx1:
                attention_probs_positive_pooled[:,:,i][condition1] = 0
        
        attention_probs_positive_pooled[:,:,idx2][condition2] = attn_sum_except_eos[condition2]
        for i in range(eos_idx):
            if i != idx2:
                attention_probs_positive_pooled[:,:,i][condition2] = 0
        
        attention_probs_positive_pooled_reshape = attention_probs_positive_pooled.reshape(16, int(32/kernel), int(32/kernel), -1)
        new_height = attention_probs_positive_pooled_reshape.shape[1] * kernel  # Since kernel_size=(2, 2)
        new_width = attention_probs_positive_pooled_reshape.shape[2] * kernel

        upsampled = F.upsample(attention_probs_positive_pooled_reshape.permute(0, 3, 1, 2), size=(new_height, new_width), mode='nearest')
        upsampled_result = upsampled.permute(0, 2, 3, 1)
        upsampled_result = upsampled_result.reshape(16, 1024, -1)
        attention_probs[16:] = upsampled_result

        return attention_probs

        
    def get_attention_scores(self,
            query: torch.Tensor, key: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, scale=None,idx1 = None,idx2 = None, eos_idx=None, timestep=None,
        ) -> torch.Tensor:

            dtype = query.dtype

            if attention_mask is None:
                baddbmm_input = torch.empty(
                    query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
                )
                beta = 0
            else:
                baddbmm_input = attention_mask
                beta = 1

            attention_scores = torch.baddbmm(
                baddbmm_input,
                query,
                key.transpose(-1, -2),
                beta=beta,
                alpha=scale,
            )
            del baddbmm_input
            
            attention_probs = attention_scores.softmax(dim=-1)
            
            if (idx1 is not None) and (idx2 is not None):
                attention_probs = self.correct_it(attention_probs=attention_probs, idx1 = idx1, idx2 = idx2, eos_idx=eos_idx, timestep=timestep)
            
            attention_probs = attention_probs.to(dtype)

            return attention_probs


    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads


        query = attn.head_to_batch_dim(query) 
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        batch = 1 if self.positive_prompt else 0

        if attention_mask is not None:
            mask_shape = attention_mask.shape
            attention_mask = attention_mask.view(mask_shape[0]*mask_shape[1], 1, -1) 


        if kwargs['kwargs']['timestep'].item() > 0:
            if query.shape == key.shape:
                attention_probs = attn.get_attention_scores(query, key, attention_mask)
            else:
                attention_probs = self.get_attention_scores(query, key, attention_mask, attn.scale,idx2=self.idx2, idx1=self.idx1, eos_idx=self.eos_idx, timestep=kwargs['kwargs']['timestep'].item()) 
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)

        
        shapes = attention_probs.shape
        attn_re = attention_probs.reshape(batch_size, attn.heads, shapes[-2], shapes[-1])
            
        ######_x
        self.attn_data_x = torch.mean(attn_re[batch], dim=0)
        ######_x
        
        hidden_states = torch.bmm(attention_probs, value) #[32, 1024, 72]
        hidden_states = attn.batch_to_head_dim(hidden_states) # [2, 1024, 1152]
        
        hidden_states = hidden_states.to(query.dtype)
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


















class AttnProcessorX:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self,positive_prompt:bool = True,idx1=[],idx2=[]):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.positive_prompt = positive_prompt

    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)

        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        batch = 1 if self.positive_prompt else 0

        if attention_mask is not None:
            mask_shape = attention_mask.shape
            attention_mask = attention_mask.view(mask_shape[0]*mask_shape[1], 1, -1) 

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        shapes = attention_probs.shape
        attn_re = attention_probs.reshape(batch_size, attn.heads, shapes[-2], shapes[-1])

        attention_probs = attn_re.reshape(batch_size* attn.heads, shapes[-2], shapes[-1])


        hidden_states = torch.bmm(attention_probs, value)


        hidden_states = attn.batch_to_head_dim(hidden_states)

        ######_x
        if attention_probs.size()[-1]==120:
            self.attn_data_x = torch.mean(attn_re[batch],dim=0)
        ######_x


        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)


        # attn.residual_connection = True
        if attn.residual_connection:
        
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
