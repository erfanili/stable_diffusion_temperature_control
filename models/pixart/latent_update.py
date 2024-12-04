


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
# from models.pixart.t5_attention_x import T5AttentionX
# from models.processors import AttnProcessor3, AttnProcessorX
from pytorch_metric_learning import distances, losses
import yaml
        
import yaml
from typing import List, Optional, Tuple

class LatentUpdateConfig:
    def __init__(self, config_path: str):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        self.max_iter_to_update: int = config.get("max_iter_to_alter", 25)
        self.refinement_steps: int = config.get("refinement_steps", 20)
        self.iterative_refinement_steps: List[int] = config.get("iterative_refinement_steps", [0, 10, 20])
        self.scale_factor: int = config.get("scale_factor", 20)
        self.attn_res: Optional[Tuple[int, int]] = tuple(config.get("attn_res", (16, 16)))
        self.steps_to_save_attention_maps: Optional[List[int]] = config.get("steps_to_save_attention_maps", None)
        self.do_smoothing: bool = config.get("do_smoothing", True)
        self.smoothing_kernel_size: int = config.get("smoothing_kernel_size", 3)
        self.smoothing_sigma: float = config.get("smoothing_sigma", 0.5)
        self.temperature: float = config.get("temperature", 0.5)
        self.softmax_normalize: bool = config.get("softmax_normalize", True)
        self.softmax_normalize_attention_maps: bool = config.get("softmax_normalize_attention_maps", False)
        self.add_previous_attention_maps: bool = config.get("add_previous_attention_maps", True)
        self.previous_attention_map_anchor_step: Optional[int] = config.get("previous_attention_map_anchor_step", None)
        self.loss_fn: str = config.get("loss_fn", "ntxent")
        self.conform: bool = config.get("conform", False)
   
   
   
   
   
   
class LatentUpdatePixartX():
    def __init__(self, config:LatentUpdateConfig):
        self.config = config
        
    @staticmethod
    def _compute_self_attn_loss_cos_attn_like(self,
        attention_maps: torch.Tensor,
        avg_text_sa_norm = None,

    ) -> torch.Tensor:
        """Computes the cosine similarity loss using the self attention of text encoder and cross attention maps."""

        attention_for_text = attention_maps[:, :, 1:-1]

        if self.config.softmax_normalize:
            attention_for_text *= 100
            attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # attention_for_text_t_plus_one = None
        # if attention_maps_t_plus_one is not None:
        #     attention_for_text_t_plus_one = attention_maps_t_plus_one[:, :, 1:-1]
        #     if self.config.softmax_normalize:
        #         attention_for_text_t_plus_one *= 100
        #         attention_for_text_t_plus_one = torch.nn.functional.softmax(
        #             attention_for_text_t_plus_one, dim=-1
        #         )
        
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
                if self.config.do_smoothing:
                    smoothing = GaussianSmoothing(
                        kernel_size=self.config.smoothing_kernel_size, sigma=self.config.smoothing_sigma
                    ).to(attention_for_text.device)
                    input = F.pad(
                        embedding_1.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect"
                    )
                    embedding_1 = smoothing(input).squeeze(0).squeeze(0)
                    
                    smoothing = GaussianSmoothing(
                        kernel_size=self.config.smoothing_kernel_size, sigma=self.config.smoothing_sigma
                    ).to(attention_for_text.device)
                    input = F.pad(
                        embedding_2.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect"
                    )
                    embedding_2 = smoothing(input).squeeze(0).squeeze(0)
                embedding_1 = embedding_1.view(-1)
                embedding_2 = embedding_2.view(-1)
                if self.config.softmax_normalize_attention_maps:
                    embedding_1 *= 100
                    embedding_1 = torch.nn.functional.softmax(embedding_1)
                    embedding_2 *= 100
                    embedding_2 = torch.nn.functional.softmax(embedding_2)
                    
                embedding_1 = embedding_1.to(attention_for_text.device)
                embedding_2 = embedding_2.to(attention_for_text.device)
                
                if self.config.cos1_or_cos2 == 'cos2':
                    cos_score = cos(embedding_1, embedding_2)
                elif self.config.cos1_or_cos2 == 'cos1':
                    norm1_score = torch.inner(embedding_1, embedding_2) / (torch.norm(embedding_1, p=1) * torch.norm(embedding_2, p=1))
                    cos_score = norm1_score*len(embedding_1)
                cos_matrix[row_idx, column_idx] = cos_score
        # breakpoint()
        attn_like_mat = cos_matrix/torch.sum(cos_matrix, dim=1, keepdim=True)
        cos_loss_lst = []
        for row_idx in range(1,text_token_len):

            if self.config.loss_type == 'cos_loss':
                cos_loss += (self.config.row_weight*(row_idx+1)/text_token_len)*(1-cos(avg_text_sa_norm[row_idx, :row_idx+1], attn_like_mat[row_idx, :row_idx+1]))
            elif self.config.loss_type == 'abs_loss':
                cos_loss += (self.config.row_weight*(row_idx+1)/text_token_len)*torch.sum(torch.abs(avg_text_sa_norm[row_idx, :row_idx+1]-attn_like_mat[row_idx, :row_idx+1]))

        return cos_loss



    def _compute_self_attn_loss_cos(self,
        attention_maps: torch.Tensor,
        avg_text_sa_norm = None,
    ) -> torch.Tensor:
        """Computes the cosine similarity loss using the self attention of text encoder and cross attention maps."""

        attention_for_text = attention_maps[:, :, 1:-1]

        if self.config.softmax_normalize:
            attention_for_text *= 100
            attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # attention_for_text_t_plus_one = None
        # if attention_maps_t_plus_one is not None:
        #     attention_for_text_t_plus_one = attention_maps_t_plus_one[:, :, 1:-1]
        #     if self.config.softmax_normalize:
        #         attention_for_text_t_plus_one *= 100
        #         attention_for_text_t_plus_one = torch.nn.functional.softmax(
        #             attention_for_text_t_plus_one, dim=-1
        #         )
        
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
                if self.config.do_smoothing:
                    smoothing = GaussianSmoothing(
                        kernel_size=self.config.smoothing_kernel_size, sigma=self.config.smoothing_sigma
                    ).to(attention_for_text.device)
                    input = F.pad(
                        embedding_1.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect"
                    )
                    embedding_1 = smoothing(input).squeeze(0).squeeze(0)
                    
                    smoothing = GaussianSmoothing(
                        kernel_size=self.config.smoothing_kernel_size, sigma=self.config.smoothing_sigma
                    ).to(attention_for_text.device)
                    input = F.pad(
                        embedding_2.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect"
                    )
                    embedding_2 = smoothing(input).squeeze(0).squeeze(0)
                embedding_1 = embedding_1.view(-1)
                embedding_2 = embedding_2.view(-1)
                if self.config.softmax_normalize_attention_maps:
                    embedding_1 *= 100
                    embedding_1 = torch.nn.functional.softmax(embedding_1)
                    embedding_2 *= 100
                    embedding_2 = torch.nn.functional.softmax(embedding_2)
                    
                embedding_1 = embedding_1.to(attention_for_text.device)
                embedding_2 = embedding_2.to(attention_for_text.device)
                
                cos_score = cos(embedding_1, embedding_2)
                cos_matrix[row_idx, column_idx] = cos_score
        # breakpoint()
        cos_loss_lst = []
        for row_idx in range(1,text_token_len):
            cos_loss += (0.2*row_idx)*(1-cos(avg_text_sa_norm[row_idx, :row_idx], cos_matrix[row_idx, :row_idx]))

        return cos_loss


    @staticmethod
    def _update_latent(self,
        latents: torch.Tensor, loss: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), [latents], retain_graph=True
        )[0]
        latents = latents - step_size * grad_cond
        return latents

    

    def perform_iterative_refinement_step_with_attn(
        self,
        unet,
        latents: torch.Tensor,
        loss: torch.Tensor,
        text_embeddings: torch.Tensor,
        attention_fetch,
        timestep: int,

    ):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent code
        according to our loss objective until the given threshold is reached for all tokens.
        """
        cnt = 0 
        loss = 100.0
        # manually set
        # loss_threshold = {5:0.005*15, 6:0.03*21, 7:0.03*28} # 0.03 is average cosine distance
        # loss_threshold = {5:0.14*4, 6:0.14*5, 7:0.14*6} # 0.03 is average cosine distance
        # loss_threshold = {5:0.005*15, 6:0.03*21, 7:0.18*6}
        for cnt in range(20):
            # print("cnt", cnt)
            # if cnt == 20 and loss > loss_threshold[eos_idx-1]:
                # print("break!")
                # break
            
        # while loss > loss_threshold[eos_idx-1] and cnt < 30:
        #     cnt += 1
        #     print(cnt)
        
            
        # for iteration in range(refinement_steps):
        #     iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            unet(latents, timestep, encoder_hidden_states=text_embeddings).sample
            unet.zero_grad()

            # Get max activation value for each subject token
            attention_fetch.store_attn_by_timestep(timestep,unet)
            attention_maps = attention_fetch.storage

            # if attn_like_loss:
            #     loss = self._compute_self_attn_loss_cos_attn_like(
            #         attention_maps=attention_maps,
            #         avg_text_sa_norm=avg_text_sa_norm,

            #     )
            # else:
            #     loss = self._compute_self_attn_loss_cos(
            #         attention_maps=attention_maps,
            #         avg_text_sa_norm=avg_text_sa_norm,
            #     )

            # if loss != 0:
            #     latents = self._update_latent(latents, loss, step_size)
            latent = self.optimize(latent = latent,
                                   text_embeddings=text_embeddings,
                                   timestep=timestep,
                                   )

        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        _ = self.unet(latents, timestep, encoder_hidden_states=text_embeddings).sample
        self.unet.zero_grad()

        # Get max activation value for each subject token
        attention_fetch.store_attn_by_timestep(timestep,unet)
        attention_maps = attention_fetch.storage

        if self.config.attn_like_loss:
            loss = self._compute_self_attn_loss_cos_attn_like(
                attention_maps=attention_maps,
                avg_text_sa_norm=text_embeddings,
         
            )
        else:
            loss = self._compute_self_attn_loss_cos(
                attention_maps=attention_maps,
                avg_text_sa_norm=text_embeddings,
            )
            
        return loss, latents
    
    
    

    def optimize(self,
                 latent,
                 attn_map,
                 text_embeddings,
                 timestep,
                 ):
        

        if self.config.attn_like_loss:
            loss = self._compute_self_attn_loss_cos_attn_like(
            attention_maps=attn_map,
            avg_text_sa_norm = text_embeddings,
        )
        else:
            loss = self._compute_self_attn_loss_cos(
                attention_maps=attn_map,
                avg_text_sa_norm = text_embeddings,
            )
        # if self.config.iterative_refinement_steps:
        #     loss, latent = self._perform_iterative_refinement_step_with_attn(
        #         timestep = timestep,
        #         unet = unet
        #         attention_maps=attn_map,
        #         avg_text_sa_norm = text_embeddings,
        #     )

        if timestep < self.config.max_iter_to_alter:
            if loss != 0:
                print("update latent")
                latent = self._update_latent(
                    latents=latent,
                    loss=loss,
                    step_size=self.config.step_size[timestep],
                )
    
        return latent


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