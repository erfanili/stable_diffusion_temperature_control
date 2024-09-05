from transformers import T5EncoderModel
from diffusers import PixArtAlphaPipeline
import torch
pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16)
pipe.to("cuda:3")