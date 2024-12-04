import json
import sys
from dataclasses import dataclass
from pathlib import Path

import clip
import numpy as np
import pyrallis
import torch
from PIL import Image
from tqdm import tqdm
import os

sys.path.append(".")
sys.path.append("..")

from utils.load_save_utils import *
from metrics.imagenet_utils import get_embedding_for_prompt, imagenet_templates




import json
import sys
from dataclasses import dataclass
from pathlib import Path

import clip
import numpy as np
import pyrallis
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess


from metrics.imagenet_utils import get_embedding_for_prompt, imagenet_templates


@dataclass
class EvalConfig:
    output_path: Path = Path("./outputs/")
    metrics_save_path: Path = Path("./metrics/")

    def __post_init__(self):
        self.metrics_save_path.mkdir(parents=True, exist_ok=True)
        
        
def aggregate_text_similarities(result_dict):
    all_averages = [result_dict[prompt]['text_similarities'] for prompt in result_dict]
    all_averages = np.array(all_averages).flatten()
    total_average = np.average(all_averages)
    total_std = np.std(all_averages)
    return total_average, total_std





def blip_text_similarity(prompt,
                          image,
                            model,
                            blip_model,
                            vis_processors,
                            device
                           ):
    with torch.no_grad():
        # extract prompt embeddings
        prompt_features = get_embedding_for_prompt(model, prompt, templates=imagenet_templates)

        # extract blip captions and embeddings

        img = vis_processors["eval"](image).to(device).unsqueeze(0)
        text = blip_model.generate({"image": img})[0]

        t = clip.tokenize([text]).to(device)
        embedding = model.encode_text(t)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        text_similarities = (embedding.float() @ prompt_features).item()

        return  {
            'text_similarities': text_similarities,
            'captions': text,
            'prompt': prompt,
        }



def aggregate_by_min_half(d):
    """ Aggregate results for the minimum similarity score for each prompt. """
    min_per_half_res = [[min(d[prompt]["first_half"], d[prompt]["second_half"])] for prompt in d]
    min_per_half_res = np.array(min_per_half_res).flatten()
    return np.average(min_per_half_res)

def aggregate_by_full_text(d):
    """ Aggregate results for the full text similarity for each prompt. """
    full_text_res = [v['full_text'] for v in d.values()]
    full_text_res = np.array(full_text_res).flatten()
    return np.average(full_text_res)
   
def clip_score(image,prompt,model,preprocess,device):
    config = EvalConfig()
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # split prompt into first and second halves
        if ' and ' in prompt:
            prompt_parts = prompt.split(' and ')
        elif ' with ' in prompt:
            prompt_parts = prompt.split(' with ')
        else:
            print(f"Unable to split prompt: {prompt}. "
                    f"Looking for 'and' or 'with' for splitting! Skipping!")
        # extract texture features
        full_text_features = get_embedding_for_prompt(model, prompt, templates=imagenet_templates)
        first_half_features = get_embedding_for_prompt(model, prompt_parts[0], templates=imagenet_templates)
        second_half_features = get_embedding_for_prompt(model, prompt_parts[1], templates=imagenet_templates)
        # extract image features
        feats = model.encode_image(image)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        # compute similarities
        full_text_similarities = (feats.float() @ full_text_features).item()
        first_half_similarities = (feats.float() @ first_half_features).item()
        second_half_similarities = (feats.float() @ second_half_features).item()

        result = {
            'full_text': full_text_similarities,
            'first_half': first_half_similarities,
            'second_half': second_half_similarities,
            'prompt': prompt,
        }
        
    return result




if __name__ == '__main__':
    device = torch.device("cuda:0")
    model, preprocess = clip.load("ViT-B/16", device)
    model.eval()
    prompt_list = []
    image_dir = './gen_images'
    methods = ['Conform', 'Linguistic_Binding', 'ours_SA_guide','SD']
    # methods = ['Conform']
    categories = ['animals','objects','animals_objects']

    blip_model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco",
                                                            is_eval=True, device=device)     


    for method in methods:
        for category in categories:
            path = os.path.join(image_dir,method,category)
            if os.path.isdir(path):
                files  = os.listdir(path)
                
                clip_score_by_prompt = {}
                text_sim_by_prompt = {}
                for file in files:
                    file_path = os.path.join(path,file)
                    image = Image.open(file_path)
                    prompt = file.split('_')[0]
                    clip_score_by_prompt[prompt] = clip_score(image = image,
                                                                prompt=prompt,
                                                                device=device,
                                                                model=model,
                                                                preprocess=preprocess)
                    
                    text_sim_by_prompt[prompt] = blip_text_similarity(image = image,
                                                                    prompt = prompt,
                                                                    device = device,
                                                                    model = model,
                                                                    blip_model = blip_model,
                                                                    vis_processors = vis_processors)
                    
                global_clip_score = {
                    'full_text_aggregation': aggregate_by_full_text(clip_score_by_prompt),
                    'min_first_second_aggregation': aggregate_by_min_half(clip_score_by_prompt),}
                
                total_average, total_std = aggregate_text_similarities(text_sim_by_prompt)
                text_sim = {
                    'average_similarity': total_average,
                    'std_similarity': total_std,
                }



                
                ###save_clip
                save_json(data=clip_score_by_prompt,directory=f'evaluation/{method}/{category}',file_name='clip_raw_metrics')
                save_json(data = global_clip_score, directory=f'evaluation/{method}/{category}', file_name='clip_aggregated_metrics.json')

                # ##save blip
                save_json(data=text_sim_by_prompt,directory=f'evaluation/{method}/{category}',file_name= "blip_raw_metrics.json")
                save_json(data=text_sim,directory=f'evaluation/{method}/{category}',file_name= "blip_aggregated_metrics.json")
