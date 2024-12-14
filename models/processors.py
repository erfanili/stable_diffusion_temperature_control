import torch
import torch.nn.functional as F 
from typing import Optional,List, Dict
from diffusers.utils import deprecate, logging
from diffusers.models.attention_processor import Attention
import math
import numpy as  np 
import matplotlib.pyplot as plt
from pytorch_metric_learning import distances, losses




import torch
import torch.nn.functional as F 
from typing import Optional
from diffusers.utils import deprecate, logging
from diffusers.models.attention_processor import Attention
import math
import torch.nn as nn



def get_processor_class(processor_name):
    processor_classes = {
        'processor_x': AttnProcessorX,
        'processor_3': AttnProcessor3,
        'processor_x_2': AttnProcessorX_2,
        'processor_x_conform': AttnProcessorXCONFORM,
        'processor_avg': AttnProcessorAvg
        
}

    return processor_classes[processor_name]







class AttnProcessorX:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self,positive_prompt:bool = True):
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
        # breakpoint()
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

        # if attention_probs.size()[-1] not in [120,77]:
        self.attn_data_x = torch.mean(attn_re[batch],dim = 0)
            
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






class AttnProcessorX_2:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self,positive_prompt:bool = True,idx1=[],idx2=[], eos_idx=None):
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
        
        # batch = 1 if self.positive_prompt else 0
        # print(query.size())
        # exit()
        if attention_mask is not None:
            mask_shape = attention_mask.shape
            attention_mask = attention_mask.view(mask_shape[0]*mask_shape[1], 1, -1) 

        # print(query.size())
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        shapes = attention_probs.shape
        attn_re = attention_probs.reshape(batch_size, attn.heads, shapes[-2], shapes[-1])

        attention_probs = attn_re.reshape(batch_size* attn.heads, shapes[-2], shapes[-1])


        hidden_states = torch.bmm(attention_probs, value)


        

        
        if attention_probs.size()[-1] in [120,77]:
            
            def get_probs(probs,value):
                # breakpoint()
                new_probs = torch.clone(probs)
                
                new_probs[:,:,1:] = torch.mean(probs[:,:,1:],dim=-1,keepdim = True)
                hidden_states = torch.bmm(new_probs, value)
                
                return hidden_states

            hidden_states = get_probs(probs=attention_probs,value = value) 
            self.attn_data_x = torch.mean(attn_re[0],dim=0)  
        hidden_states = attn.batch_to_head_dim(hidden_states)
            
            # torch.Size([1, #head, #query_flatten, 77])
            # for checking
            # print("key shape", key.shape)
            # self.key_store = key.reshape(1, 77, -1)



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


    def correct_it(self,attention_probs:torch.Tensor, num_heads,idx1, idx2, eos_idx, timestep:int, alpha1:float=2, alpha2:float=3)-> torch.Tensor:

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
        map_size = int(math.sqrt(attention_probs.size(1)))
  
        attention_probs_positive = attention_probs[num_heads:]

        kernel = 1
        attention_probs_positive_reshape = attention_probs_positive.reshape(num_heads, map_size, map_size, -1)
        tensor_permuted = attention_probs_positive_reshape.permute(0, 3, 1, 2) # num_heads, 120, map_size, map_size
        pooled = F.avg_pool2d(tensor_permuted, kernel_size=(kernel, kernel))  # num_heads, 120, num_heads, num_heads
        attention_probs_positive_pooled = pooled.permute(0, 2, 3, 1) # num_heads, num_heads, num_heads, 120
        attention_probs_positive_pooled = attention_probs_positive_pooled.reshape(num_heads, int(map_size**2/(kernel*kernel)), -1)
        
        attn_scr1 = attention_probs_positive_pooled[:,:,idx1]
        attn_scr2 = attention_probs_positive_pooled[:,:,idx2]
    
        condition1 = attn_scr1 > attn_scr2*1.3
        condition2 = attn_scr1*1.3 <= attn_scr2

        if (torch.sum(condition1) < 2400 or torch.sum(condition2) < 2400) and timestep > 800:
            
            noise = torch.rand(1, 1, int(map_size/kernel), int(map_size/kernel))  # Shape: (batch_size, channels, height, width)
            gaussian_kernel = torch.tensor([[[[1, 2, 1], [1, 1, 2], [2, 1, 1]]]], dtype=torch.float32)
            gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
            smoothed_noise = F.conv2d(noise, gaussian_kernel, padding=1)
            mask = smoothed_noise > 0.5 
            binary_mask = mask.squeeze(0).squeeze(0)
            condition1 = binary_mask.flatten()
            condition2 = ~binary_mask.flatten()
            condition1 = condition1.unsqueeze(0).repeat(num_heads, 1)
            condition2 = condition2.unsqueeze(0).repeat(num_heads, 1)
            
        attn_sum_except_eos = torch.sum(attention_probs_positive_pooled[:,:,:eos_idx], axis=-1)

        attention_probs_positive_pooled[:,:,idx1][condition1] = attn_sum_except_eos[condition1]
        for i in range(eos_idx):
            if i != idx1:
                attention_probs_positive_pooled[:,:,i][condition1] = 0
        
        attention_probs_positive_pooled[:,:,idx2][condition2] = attn_sum_except_eos[condition2]
        for i in range(eos_idx):
            if i != idx2:
                attention_probs_positive_pooled[:,:,i][condition2] = 0
        
        attention_probs_positive_pooled_reshape = attention_probs_positive_pooled.reshape(num_heads, int(map_size/kernel), int(map_size/kernel), -1)
        new_height = attention_probs_positive_pooled_reshape.shape[1] * kernel  # Since kernel_size=(2, 2)
        new_width = attention_probs_positive_pooled_reshape.shape[2] * kernel

        upsampled = F.upsample(attention_probs_positive_pooled_reshape.permute(0, 3, 1, 2), size=(new_height, new_width), mode='nearest')
        upsampled_result = upsampled.permute(0, 2, 3, 1)
        upsampled_result = upsampled_result.reshape(num_heads, map_size**2, -1)

        attention_probs[num_heads:] = upsampled_result

        return attention_probs


                
    def get_probs(self,probs,value):
        # breakpoint()
        new_probs = torch.clone(probs)
        
        new_probs[:,:,1:] = torch.mean(probs[:,:,1:],dim=-1,keepdim = True)

        return hidden_states

            # hidden_states = get_probs(probs=attention_probs,value = value)  
    def get_attention_scores(self,
            query: torch.Tensor, key: torch.Tensor, num_heads:int,attention_mask: Optional[torch.Tensor] = None, scale=None,idx1 = None,idx2 = None, eos_idx=None, timestep=None,
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
            # breakpoint()
            attention_scores[:,:,1:] = torch.mean(attention_scores[:,:,1:],dim = -1,keepdim= True)
            attention_probs = attention_scores.softmax(dim=-1)
            
            
            
            # if key.size() != query.size():
                # print(torch.mean(attention_probs[8:,:,0]),torch.std(attention_probs[8:,:,0]))
                # eta  = 1/(attention_probs[12,:,0])-1
                # eps = torch.median(eta)
                # if eps <.8:
                #     large = torch.sum(torch.where(eta>.8,1,0)).item()
                #     if large != 0:
                # torch.set_printoptions(precision=2)

                # print(f'mean: {torch.mean(query[12,:,1]).item():.2f}, std: {torch.std(query[12,:,1]).item():.2f}')
                    # eta = eta.detach().cpu().numpy()
                    # plt.hist(eta.flatten(),bins = 50)
                    # # plt.imshow()
                    # plt.savefig('eta_hist.png')
                    # exit()
                    # print(self._get_object_chain())
                # print(eps)
            # if (idx1 is not None) and (idx2 is not None):
            #     attention_probs = self.correct_it(attention_probs=attention_probs,num_heads=num_heads, idx1 = idx1, idx2 = idx2, eos_idx=eos_idx, timestep=timestep)
            
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
        # breakpoint()
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
                # key[:,1:,:] = torch.mean(key[:,1:,:],dim = 1,keepdim = True)
                attention_probs = self.get_attention_scores(query=query, key=key, attention_mask=attention_mask,num_heads=attn.heads, scale = attn.scale,idx2=self.idx2, idx1=self.idx1, eos_idx=self.eos_idx, 
                                                            timestep=kwargs['kwargs']['timestep'].item()) 

        else:

            attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # attention_probs = attn.get_attention_scores(query, key, attention_mask)

        shapes = attention_probs.shape
        attn_re = attention_probs.reshape(batch_size, attn.heads, shapes[-2], shapes[-1])
            
        ######_x

        self.attn_data_x = torch.mean(attn_re[batch], dim=0)

        ######_x
        
        hidden_states = torch.bmm(attention_probs, value) #[map_size, 1024, 72]
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

















class AttnProcessor4:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self,emb_sim:torch.Tensor,positive_prompt:bool = True):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.positive_prompt = positive_prompt
        self.emb_sim = emb_sim
    
    
    
        
    def get_attention_scores(self,
                query: torch.Tensor, key: torch.Tensor, num_heads:int,attention_mask: Optional[torch.Tensor] = None, scale=None,idx1 = None,idx2 = None, eos_idx=None, timestep=None,
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
            alpha=1/5,
        )
        del baddbmm_input
        
        
        
        #####
                # np.set_printoptions(suppress=True, precision=2)

        # Extract tensor slice and normalize it
        X = attention_scores[ 8:, :, 1:10]
        
        self.emb_sim = self.emb_sim.to(query.device)
        
        def process_map(attn,emb_sim):
            dtype = attn.dtype
            attn = attn.to(torch.float64)
            q_norms = torch.norm(attn, dim=1,p=2, keepdim=True)
            k_norms = torch.norm(attn,dim=2, p =1, keepdim = True)
            attn /=q_norms
            
            attn_square = torch.bmm(attn.transpose(1,2) , attn)
            
            _, s, Vh = torch.linalg.svd(attn_square)
            # print(s[0])
            # exit()
            
            p = Vh.transpose(1,2) @torch.sqrt(s).diag_embed() @ Vh
            np.set_printoptions(suppress=True, precision=1)
            # for i in range(8):
                # print('p', p[i].detach().cpu().numpy())
            # print('a2', attn_square[0].shape)
            # print(key.size())
            p_inv = torch.inverse(p)
            new_attn = attn @ p_inv

            new_attn = new_attn.to(torch.float32)
            _,sigma, nu = torch.linalg.svd(emb_sim)
            
            new_p = sigma.diag_embed() @ nu
            new_attn = new_attn @ new_p

            attn = new_attn.to(dtype)

            return attn
        
        Y = process_map(attn = X, emb_sim = self.emb_sim)
        
        attention_scores[ 8:, :, 1:10]  = Y
        
        
        
        #####
        attention_probs = attention_scores.softmax(dim=-1)
        
        # if (idx1 is not None) and (idx2 is not None):
        #     attention_probs = self.correct_it(attention_probs=attention_probs,num_heads=num_heads, idx1 = idx1, idx2 = idx2, eos_idx=eos_idx, timestep=timestep)
        
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

        # print(key.size())
        ######_x
        if key.size()[1] in [120,77]:

            attention_probs = self.get_attention_scores(query, key, attention_mask)

            
        ######_x
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)

        shapes = attention_probs.shape
        attn_re = attention_probs.reshape(batch_size, attn.heads, shapes[-2], shapes[-1])

        self.attn_data_x = torch.mean(attn_re[batch],dim=0)
        # attention_probs = attn_re.reshape(batch_size* attn.heads, shapes[-2], shapes[-1])

        hidden_states = torch.bmm(attention_probs, value)


        hidden_states = attn.batch_to_head_dim(hidden_states)
        
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










class AttnProcessorXCONFORM:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self,positive_prompt:bool = True,
                 ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.positive_prompt = positive_prompt


        
    def get_attention_scores(self,
            query: torch.Tensor, key: torch.Tensor, 
            num_heads:int,
            attention_mask: Optional[torch.Tensor] = None, 
            scale=None,

            
            timestep=None,
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
            
            # if (idx1 is not None) and (idx2 is not None):
            #     attention_probs = self.correct_it(attention_probs=attention_probs,
            #                                       num_heads=num_heads, idx1 = idx1, idx2 = idx2, eos_idx=eos_idx, timestep=timestep)
            
            attention_probs = attention_probs.to(dtype)

            return attention_probs




    @staticmethod
    def _update_query(
        latents: torch.Tensor, loss: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), [latents], retain_graph=True
        )[0]
        latents = latents - step_size * grad_cond
        return latents
    
    
    

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        attention_map_t_plus_one = None,
        avg_block_text_sa_norm = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        max_iter_to_alter = 30,
        token_group = [[2,3], [6,7]],
        num_images_per_prompt = 1,
        num_inference_steps = 20,
        use_conform = True,
        refinement_steps: int = 20,
        loss_fn = "ntxent",
        temperature = 0.5,
        do_smoothing: bool = True,
        smoothing_kernel_size: int = 3,
        smoothing_sigma: float = 0.5,
        softmax_normalize: bool = True,
        softmax_normalize_attention_maps: bool = False,
        iterative_refinement_steps: List[int] = [0, 10, 20],
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

        # print(query.size())

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        batch = 1 if self.positive_prompt else 0

        if attention_mask is not None:
            mask_shape = attention_mask.shape
            attention_mask = attention_mask.view(mask_shape[0]*mask_shape[1], 1, -1) 

        timestep = kwargs['kwargs']['timestep'].item()
        step_size = kwargs['kwargs']['step_size']
        # attention_probs = attn.get_attention_scores(query, key, attention_mask)
        if query.shape == key.shape:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            
        else:
            num_steps = max_iter_to_alter if timestep in iterative_refinement_steps else 1

            for _ in range(num_steps+1):
                # num_steps+1 because we need 2 iterations if timestep is not [0 , 10, 20] to get one step attention update
                attention_probs = self.get_attention_scores(query=query, 
                                                        key=key, 
                                                        attention_mask=attention_mask,
                                                        num_heads=attn.heads, 
                                                        scale = attn.scale,
            
                                                        # eos_idx=self.eos_idx, 
                                                        timestep=timestep) 

                if use_conform:
                    _probs = torch.mean(attention_probs, dim=0).reshape(16,16,-1)
                    loss = self._compute_contrastive_loss(
                        attention_maps=_probs,
                        attention_maps_t_plus_one=attention_map_t_plus_one,
                        token_groups=token_group,
                        loss_type=loss_fn,
                        temperature=temperature,
                        do_smoothing=do_smoothing,
                        smoothing_kernel_size=smoothing_kernel_size,
                        smoothing_sigma=smoothing_sigma,
                        softmax_normalize=softmax_normalize,
                        softmax_normalize_attention_maps=softmax_normalize_attention_maps,
                    )
                # else:
                #     loss = self._compute_self_attn_loss_cos(
                #         attention_maps=attention_probs,
                #         attention_maps_t_plus_one=attention_map_t_plus_one,
                #         do_smoothing=do_smoothing,
                #         smoothing_kernel_size=smoothing_kernel_size,
                #         smoothing_sigma=smoothing_sigma,
                #         softmax_normalize=softmax_normalize,
                #         softmax_normalize_attention_maps=softmax_normalize_attention_maps,
                #         # self_attn_score_pairs=self_attn_score_pairs,
                #         avg_text_sa_norm = avg_block_text_sa_norm,
                #     )                    
                            
                if loss != 0:
                    query = self._update_query(query, loss, step_size)

                  
        shapes = attention_probs.shape
        # print(shapes)
        # exit()
        attn_re = attention_probs.reshape(batch_size, attn.heads, shapes[-2], shapes[-1])

        attention_probs = attn_re.reshape(batch_size* attn.heads, shapes[-2], shapes[-1])


        hidden_states = torch.bmm(attention_probs, value)


        hidden_states = attn.batch_to_head_dim(hidden_states)

        ######_x
        if attention_probs.size()[-1] in [120,77]:
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
    
    def _compute_contrastive_loss(self,
            attention_maps: torch.Tensor,
            attention_maps_t_plus_one: Optional[torch.Tensor],
            token_groups: List[List[int]],
            loss_type: str,
            temperature: float = 0.07,
            do_smoothing: bool = True,
            smoothing_kernel_size: int = 3,
            smoothing_sigma: float = 0.5,
            softmax_normalize: bool = True,
            softmax_normalize_attention_maps: bool = False,
        ) -> torch.Tensor:
            """Computes the attend-and-contrast loss using the maximum attention value for each token."""
            # breakpoint()
            attention_for_text = attention_maps[:, :, 1:-1]

            if softmax_normalize:
                attention_for_text *= 100
                attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

            attention_for_text_t_plus_one = None
            if attention_maps_t_plus_one is not None:
                attention_for_text_t_plus_one = attention_maps_t_plus_one[:, :, 1:-1]
                if softmax_normalize:
                    attention_for_text_t_plus_one *= 100
                    attention_for_text_t_plus_one = torch.nn.functional.softmax(
                        attention_for_text_t_plus_one, dim=-1
                    )

            indices_to_clases = {}
            for c, group in enumerate(token_groups):
                for obj in group:
                    indices_to_clases[obj] = c

            classes = []
            embeddings = []
            for ind, c in indices_to_clases.items():
                classes.append(c)
                # Shift indices since we removed the first token
                embedding = attention_for_text[:, :, ind - 1]
                if do_smoothing:
                    smoothing = GaussianSmoothing(
                        kernel_size=smoothing_kernel_size, sigma=smoothing_sigma
                    ).to(attention_for_text.device)
                    input = F.pad(
                        embedding.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect"
                    )
                    embedding = smoothing(input).squeeze(0).squeeze(0)
                embedding = embedding.view(-1)

                if softmax_normalize_attention_maps:
                    embedding *= 100
                    embedding = torch.nn.functional.softmax(embedding)
                embeddings.append(embedding)

                if attention_for_text_t_plus_one is not None:
                    classes.append(c)
                    # Shift indices since we removed the first token
                    embedding = attention_for_text_t_plus_one[:, :, ind - 1]
                    if do_smoothing:
                        smoothing = GaussianSmoothing(
                            kernel_size=smoothing_kernel_size, sigma=smoothing_sigma
                        ).to(attention_for_text.device)
                        input = F.pad(
                            embedding.unsqueeze(0).unsqueeze(0),
                            (1, 1, 1, 1),
                            mode="reflect",
                        )
                        embedding = smoothing(input).squeeze(0).squeeze(0)
                    embedding = embedding.view(-1)

                    if softmax_normalize_attention_maps:
                        embedding *= 100
                        embedding = torch.nn.functional.softmax(embedding)
                    embeddings.append(embedding)

            classes = torch.tensor(classes).to(attention_for_text.device)
            embeddings = torch.stack(embeddings, dim=0).to(attention_for_text.device)

            # loss_fn = losses.NTXentLoss(temperature=temperature)

            if loss_type == "ntxent_contrastive":
                if len(token_groups) > 0 and len(token_groups[0]) > 1:
                    loss_fn = losses.NTXentLoss(temperature=temperature)
                else:
                    loss_fn = losses.ContrastiveLoss(
                        distance=distances.CosineSimilarity(), pos_margin=1, neg_margin=0
                    )
            elif loss_type == "ntxent":
                loss_fn = losses.NTXentLoss(temperature=temperature)
            elif loss_type == "contrastive":
                loss_fn = losses.ContrastiveLoss(
                    distance=distances.CosineSimilarity(), pos_margin=1, neg_margin=0
                )
            else:
                raise ValueError(f"loss_fn {loss_type} not supported")

            loss = loss_fn(embeddings, classes)
            return loss
            
    def _compute_self_attn_loss_cos(
        attention_maps: torch.Tensor,
        attention_maps_t_plus_one: Optional[torch.Tensor],
        do_smoothing: bool = True,
        smoothing_kernel_size: int = 3,
        smoothing_sigma: float = 0.5,
        softmax_normalize: bool = True,
        softmax_normalize_attention_maps: bool = False,
        self_attn_score_pairs=None,
        avg_text_sa_norm = None,
    ) -> torch.Tensor:
        """Computes the cosine similarity loss using the self attention of text encoder and cross attention maps."""

        attention_for_text = attention_maps[:, :, 1:-1]

        if softmax_normalize:
            attention_for_text *= 100
            attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        attention_for_text_t_plus_one = None
        if attention_maps_t_plus_one is not None:
            attention_for_text_t_plus_one = attention_maps_t_plus_one[:, :, 1:-1]
            if softmax_normalize:
                attention_for_text_t_plus_one *= 100
                attention_for_text_t_plus_one = torch.nn.functional.softmax(
                    attention_for_text_t_plus_one, dim=-1
                )
        
        cos = nn.CosineSimilarity(dim=0)
        cos_loss = 0
        text_token_len = avg_text_sa_norm.shape[0]
        # make cross attn cos sim matrix
        cos_matrix = torch.eye(text_token_len, dtype=attention_for_text.dtype).to(attention_for_text.device)
        for row_idx in range(text_token_len):    
            for column_idx in range(row_idx+1):
                if row_idx == column_idx:
                    continue
                embedding_1 = attention_for_text[:, :, row_idx]
                embedding_2 = attention_for_text[:, :, column_idx]
                if do_smoothing:
                    smoothing = GaussianSmoothing(
                        kernel_size=smoothing_kernel_size, sigma=smoothing_sigma
                    ).to(attention_for_text.device)
                    input = F.pad(
                        embedding_1.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect"
                    )
                    embedding_1 = smoothing(input).squeeze(0).squeeze(0)
                    
                    smoothing = GaussianSmoothing(
                        kernel_size=smoothing_kernel_size, sigma=smoothing_sigma
                    ).to(attention_for_text.device)
                    input = F.pad(
                        embedding_2.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect"
                    )
                    embedding_2 = smoothing(input).squeeze(0).squeeze(0)
                embedding_1 = embedding_1.view(-1)
                embedding_2 = embedding_2.view(-1)
                if softmax_normalize_attention_maps:
                    embedding_1 *= 100
                    embedding_1 = torch.nn.functional.softmax(embedding_1)
                    embedding_2 *= 100
                    embedding_2 = torch.nn.functional.softmax(embedding_2)
                    
                embedding_1 = embedding_1.to(attention_for_text.device)
                embedding_2 = embedding_2.to(attention_for_text.device)
                
                cos_score = cos(embedding_1, embedding_2)
                cos_matrix[row_idx, column_idx] = cos_score
        # breakpoint()
        for row_idx in range(1,text_token_len):
            # for column_idx in range(row_idx + 1):
            cos_loss += row_idx*(1-cos(avg_text_sa_norm[row_idx, :row_idx], cos_matrix[row_idx, :row_idx]))
            # cos_loss += (1.5**row_idx)*(1-cos(avg_text_sa_norm[row_idx, :row_idx], cos_matrix[row_idx, :row_idx]))
            
        return cos_loss


    # def _perform_iterative_refinement_step(
    #     self,
    #     queries: torch.Tensor,
    #     token_groups: List[List[int]],
    #     loss: torch.Tensor,
    #     text_embeddings: torch.Tensor,
    #     step_size: float,
    #     t: int,
    #     refinement_steps: int = 20,
    #     do_smoothing: bool = True,
    #     smoothing_kernel_size: int = 3,
    #     smoothing_sigma: float = 0.5,
    #     temperature: float = 0.07,
    #     softmax_normalize: bool = True,
    #     softmax_normalize_attention_maps: bool = False,
    #     attention_maps_t_plus_one: Optional[torch.Tensor] = None,
    #     loss_fn: str = "ntxent",
    # ):
    #     """
    #     Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent code
    #     according to our loss objective until the given threshold is reached for all tokens.
    #     """
    #     for iteration in range(refinement_steps):
    #         iteration += 1

    #         queries = queries.clone().detach().requires_grad_(True)
    #         # self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
    #         # self.unet.zero_grad()

    #         # Get max activation value for each subject token
    #         # attention_maps = self._aggregate_attention()
            
    #         # self.attn_fetch_x.store_attn_by_timestep(t,self.unet)
    #         # attention_maps = self.attn_fetch_x.storage

    #         loss = self._compute_contrastive_loss(
    #             attention_maps=attention_maps,
    #             attention_maps_t_plus_one=attention_maps_t_plus_one,
    #             token_groups=token_groups,
    #             loss_type=loss_fn,
    #             do_smoothing=do_smoothing,
    #             temperature=temperature,
    #             smoothing_kernel_size=smoothing_kernel_size,
    #             smoothing_sigma=smoothing_sigma,
    #             softmax_normalize=softmax_normalize,
    #             softmax_normalize_attention_maps=softmax_normalize_attention_maps,
    #         )

    #         if loss != 0:
    #             latents = self._update_latent(latents, loss, step_size)

    #     # Run one more time but don't compute gradients and update the latents.
    #     # We just need to compute the new loss - the grad update will occur below
    #     # latents = latents.clone().detach().requires_grad_(True)
    #     # _ = self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
    #     # self.unet.zero_grad()

    #     # # Get max activation value for each subject token
    #     # self.attn_fetch_x.store_attn_by_timestep(t,self.unet)
    #     # attention_maps = self.attn_fetch_x.storage

    #     loss = self._compute_contrastive_loss(
    #         attention_maps=attention_maps,
    #         attention_maps_t_plus_one=attention_maps_t_plus_one,
    #         token_groups=token_groups,
    #         loss_type=loss_fn,
    #         do_smoothing=do_smoothing,
    #         temperature=temperature,
    #         smoothing_kernel_size=smoothing_kernel_size,
    #         smoothing_sigma=smoothing_sigma,
    #         softmax_normalize=softmax_normalize,
    #         softmax_normalize_attention_maps=softmax_normalize_attention_maps,
    #     )
    #     return loss, latents
    
    # def _perform_iterative_refinement_step_with_attn(
    #     self,
    #     latents: torch.Tensor,
    #     loss: torch.Tensor,
    #     text_embeddings: torch.Tensor,
    #     step_size: float,
    #     t: int,
    #     refinement_steps: int = 20,
    #     do_smoothing: bool = True,
    #     smoothing_kernel_size: int = 3,
    #     smoothing_sigma: float = 0.5,
    #     softmax_normalize: bool = True,
    #     softmax_normalize_attention_maps: bool = False,
    #     attention_maps_t_plus_one: Optional[torch.Tensor] = None,
    #     self_attn_score_pairs=None,
    #     avg_text_sa_norm=None,
    # ):
    #     """
    #     Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent code
    #     according to our loss objective until the given threshold is reached for all tokens.
    #     """
    #     for iteration in range(refinement_steps):
    #         iteration += 1

    #         latents = latents.clone().detach().requires_grad_(True)
    #         self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
    #         self.unet.zero_grad()

    #         # Get max activation value for each subject token
    #         self.attn_fetch_x.store_attn_by_timestep(t,self.unet)
    #         attention_maps = self.attn_fetch_x.storage

    #         loss = self._compute_self_attn_loss_cos(
    #             attention_maps=attention_maps,
    #             attention_maps_t_plus_one=attention_maps_t_plus_one,
    #             do_smoothing=do_smoothing,
    #             smoothing_kernel_size=smoothing_kernel_size,
    #             smoothing_sigma=smoothing_sigma,
    #             softmax_normalize=softmax_normalize,
    #             softmax_normalize_attention_maps=softmax_normalize_attention_maps,
    #             self_attn_score_pairs=self_attn_score_pairs,
    #             avg_text_sa_norm=avg_text_sa_norm,
    #         )

    #         if loss != 0:
    #             latents = self._update_latent(latents, loss, step_size)

    #     # Run one more time but don't compute gradients and update the latents.
    #     # We just need to compute the new loss - the grad update will occur below
    #     latents = latents.clone().detach().requires_grad_(True)
    #     _ = self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
    #     self.unet.zero_grad()

    #     # Get max activation value for each subject token
    #     self.attn_fetch_x.store_attn_by_timestep(t,self.unet)
    #     attention_maps = self.attn_fetch_x.storage

    #     loss = self._compute_self_attn_loss_cos(
    #         attention_maps=attention_maps,
    #         attention_maps_t_plus_one=attention_maps_t_plus_one,
    #         do_smoothing=do_smoothing,
    #         smoothing_kernel_size=smoothing_kernel_size,
    #         smoothing_sigma=smoothing_sigma,
    #         softmax_normalize=softmax_normalize,
    #         softmax_normalize_attention_maps=softmax_normalize_attention_maps,
    #         self_attn_score_pairs=self_attn_score_pairs,
    #         avg_text_sa_norm=avg_text_sa_norm,
    #     )
    #     return loss, latents
    # def optimize_query(self,
    #                    query,

                       
    # ):
 

        # steps_to_save_attention_maps = list(range(num_inference_steps))

   
        # if isinstance(token_groups[0][0], int):
        #     token_groups = [token_groups]
        # # attention_map = [{} for i in range(num_images_per_prompt)]

        # query = query.clone().detach().requires_grad_(True)
        # updated_query = []


            # print("attn loss", loss)

            # If this is an iterative refinement step, verify we have reached the desired threshold for all
        # if timestep in iterative_refinement_steps:
        #     if conform:
        #         loss, latent = self._perform_iterative_refinement_step(
        #             latents=latent,
        #             token_groups=token_group,
        #             loss=loss,
        #             text_embeddings=text_embedding,
        #             step_size=step_size,
        #             t=t,
        #             refinement_steps=refinement_steps,
        #             do_smoothing=do_smoothing,
        #             smoothing_kernel_size=smoothing_kernel_size,
        #             smoothing_sigma=smoothing_sigma,
        #             temperature=temperature,
        #             softmax_normalize=softmax_normalize,
        #             softmax_normalize_attention_maps=softmax_normalize_attention_maps,
        #             attention_maps_t_plus_one=attention_map_t_plus_one,
        #             loss_fn=loss_fn,
        #             )
        #     else:
        #         loss, latent = self._perform_iterative_refinement_step_with_attn(
        #             latents=latent,
        #             loss=loss,
        #             text_embeddings=text_embedding,
        #             step_size=step_size,
        #             t=t,
        #             refinement_steps=refinement_steps,
        #             do_smoothing=do_smoothing,
        #             smoothing_kernel_size=smoothing_kernel_size,
        #             smoothing_sigma=smoothing_sigma,
        #             softmax_normalize=softmax_normalize,
        #             softmax_normalize_attention_maps=softmax_normalize_attention_maps,
        #             attention_maps_t_plus_one=attention_map_t_plus_one,
        #             # self_attn_score_pairs=self_attn_score_pairs,
        #             avg_text_sa_norm = avg_block_text_sa_norm,
        #         )
            # Perform gradient updat
        #     if loss != 0:
        #         print("update latent")
        #         updated_query = self._update_query(
        #             latents=query,
        #             loss=loss,
        #             step_size=step_size,
        #         )
        #     # updated_queries.append(latent)
        # return updated_query
    


class GaussianSmoothing(torch.nn.Module):
    """
    Arguments:
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed seperately for each channel in the input
    using a depthwise convolution.
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel. sigma (float, sequence): Standard deviation of the
        gaussian kernel. dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    # channels=1, kernel_size=kernel_size, sigma=sigma, dim=2
    def __init__(
        self,
        channels: int = 1,
        kernel_size: int = 3,
        sigma: float = 0.5,
        dim: int = 2,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Arguments:
        Apply gaussian filter to input.
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups)
    
    
    
    
class AttnProcessorAvg:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self,positive_prompt:bool = True,eos_idx=[]):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")


        self.eos_idx = eos_idx
        self.positive_prompt = positive_prompt


    def get_attention_scores(self,
            query: torch.Tensor, key: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, scale=None, **kwargs
        ) -> torch.Tensor:
            dtype = query.dtype
            # breakpoint()
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
            # breakpoint()
            eos_idx = kwargs['eos_idx']
            
            # new_scores = attention_scores.clone()
            # new_scores[16:,:,:eos_idx] = torch.mean(attention_scores[16:,:,:eos_idx],dim = -1,keepdim= True)
            # breakpoint()
            attention_probs = attention_scores.softmax(dim=-1)
            # new_probs = new_scores.softmax(dim = -1)
            avg_probs = attention_probs.clone().float()
            breakpoint()
            avg_probs[16:,:,:eos_idx] =torch.mean(avg_probs[16:,:,:eos_idx],dim =-1, keepdim = True)
            
        
            attention_probs = attention_probs.to(dtype)
            avg_probs = avg_probs.to(dtype)
            return avg_probs


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
        # breakpoint()
        value = attn.head_to_batch_dim(value)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        batch = 1 if self.positive_prompt else 0

        if attention_mask is not None:
            mask_shape = attention_mask.shape
            attention_mask = attention_mask.view(mask_shape[0]*mask_shape[1], 1, -1) 



        if query.shape != key.shape and kwargs['kwargs']['timestep']>0:
            print('yes')
            attention_probs = self.get_attention_scores(query=query, key=key, attention_mask=attention_mask,num_heads=attn.heads, scale = attn.scale,
                                                         eos_idx=kwargs['kwargs']['eos_idx'])     

        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
    
            
        # else:

            # attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # attention_probs = attn.get_attention_scores(query, key, attention_mask)

        shapes = attention_probs.shape
        attn_re = attention_probs.reshape(batch_size, attn.heads, shapes[-2], shapes[-1])
            
        ######_x

        self.attn_data_x = torch.mean(attn_re[batch], dim=0)

        ######_x
        
        hidden_states = torch.bmm(attention_probs, value) #[map_size, 1024, 72]
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








