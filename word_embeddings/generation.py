import torch 
import os 
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline 
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers.models.clip.configuration_clip import CLIPConfig
import torch.nn.functional as F
from datetime import datetime 
from utils import *
import random



    
    

        
prompt_list = [
    "The cat jumped over the fence and disappeared into the night.",
    "Bright orange flowers bloomed along the garden path.",
    "She wondered if the rumors about the haunted house were true.",
    "A sudden downpour drenched everyone at the outdoor concert.",
    "He carefully placed the last piece of the puzzle into its spot.",
    "The old man sat on the park bench, feeding the pigeons.",
    "Her laughter echoed through the empty hallways of the school.",
    "The smell of freshly baked bread filled the kitchen.",
    "A lone sailboat drifted across the calm sea.",
    "The child stared in awe at the towering skyscrapers of the city.",
    "They huddled around the campfire, sharing ghost stories until midnight.",
    "The wind whispered secrets through the tall pine trees.",
    "She found an old photograph tucked inside the book's pages.",
    "The city lights twinkled like stars in the distance.",
    "A butterfly landed gently on her hand, fluttering its wings.",
    "He played the piano with a passion that brought the audience to tears.",
    "The mountain peak was covered in a blanket of snow.",
    "The clock struck midnight, and the party was still in full swing.",
    "She couldn't shake the feeling that someone was watching her.",
    "The scent of lavender filled the room, calming her nerves.",
    "The little boy chased after the ice cream truck, laughing all the way.",
    "She gazed up at the full moon, lost in thought.",
    "The train whistle echoed through the valley as it sped past.",
    "He picked up the old, worn-out book and started reading.",
    "The dog barked excitedly as the mailman approached the house.",
    "A gentle breeze rustled the leaves on the trees.",
    "She found a hidden door in the back of the closet.",
    "The sun set behind the mountains, casting a golden glow.",
    "He stared at the blank canvas, unsure of what to paint.",
    "The streets were empty, and the city was eerily quiet.",
    "She smiled as she watched the children play in the park.",
    "The waves crashed against the shore, creating a soothing sound.",
    "He held the tiny bird in his hands, feeling its rapid heartbeat.",
    "The fire crackled and popped as they roasted marshmallows.",
    "She carefully wrapped the gift in colorful paper and tied it with a bow.",
    "The stars twinkled in the night sky, far above the city lights.",
    "He opened the dusty old trunk and found it filled with treasures.",
    "The garden was alive with the buzzing of bees and the chirping of birds.",
    "She walked along the beach, feeling the sand between her toes.",
    "The old clock on the wall ticked away the minutes in the quiet room.",
    "He watched as the sun rose over the horizon, painting the sky in shades of pink and orange.",
    "She found a tiny seashell in the sand and slipped it into her pocket.",
    "The river flowed gently, reflecting the colors of the sunset.",
    "He tied the boat to the dock and jumped onto the pier.",
    "The cat curled up in the sunlight streaming through the window.",
    "She read the letter over and over, trying to understand its meaning.",
    "The thunder rumbled in the distance, signaling the approaching storm.",
    "He lit the candle and watched the flame flicker in the dark room.",
    "She placed the flowers in a vase and set them on the table.",
    "The children ran through the sprinklers, squealing with delight.",
    "He stood at the edge of the cliff, looking out at the vast ocean.",
    "The rain tapped gently on the roof, creating a soothing rhythm.",
    "She braided her hair and tied it with a ribbon.",
    "The smell of coffee filled the air as he brewed a fresh pot.",
    "The bird perched on the windowsill, singing a cheerful tune.",
    "She dipped her toes into the cool water of the lake.",
    "The streetlights flickered on as the sun set.",
    "He scribbled a note on a piece of paper and left it on the kitchen counter.",
    "She picked up the old, tattered teddy bear and hugged it tightly.",
    "The wind blew through the open window, ruffling the curtains.",
    "He stepped onto the train, ready for a new adventure.",
    "The snowflakes fell softly, covering the ground in a white blanket.",
    "She glanced at the clock and realized she was running late.",
    "The cat stretched lazily in the warm sunlight.",
    "He poured a cup of tea and sat down to read the morning paper.",
    "The leaves crunched underfoot as she walked through the forest.",
    "The sound of waves crashing against the rocks was mesmerizing.",
    "He held the umbrella over her head as they walked in the rain.",
    "She opened the window and let the fresh air fill the room.",
    "The fireflies danced in the night, their lights twinkling like stars.",
    "He wiped the sweat from his brow and continued working in the garden.",
    "The smell of freshly baked cookies wafted through the house.",
    "She spotted a rainbow in the sky after the rain.",
    "The old piano in the corner of the room was covered in dust.",
    "He watched the sunset, feeling a sense of peace wash over him.",
    "The bird soared high above the trees, its wings spread wide.",
    "She felt a sense of accomplishment as she crossed the finish line.",
    "The snow crunched underfoot as he made his way up the hill.",
    "She picked a bouquet of wildflowers and carried them home.",
    "The dog wagged its tail happily as it greeted its owner.",
    "He listened to the sound of the rain on the roof and drifted off to sleep.",
    "The moon cast a pale light on the deserted street.",
    "She found an old, forgotten diary in the attic.",
    "The waves lapped gently against the side of the boat.",
    "He watched as the clouds slowly drifted across the sky.",
    "The cat purred contentedly as it curled up in her lap.",
    "She blew out the candles on the cake and made a wish.",
    "The trees swayed in the wind, their leaves rustling softly.",
    "He threw a pebble into the lake and watched the ripples spread out.",
    "The fire in the fireplace crackled and glowed warmly.",
    "She sat on the porch swing, sipping lemonade and enjoying the summer breeze.",
    "The stars above twinkled like diamonds in the night sky.",
    "He reached the summit of the mountain and looked out at the breathtaking view.",
    "She wrote a letter to her friend and sealed it with a kiss.",
    "The car's headlights cut through the darkness as it drove down the lonely road.",
    "He watched the sunrise from the top of the hill, feeling the warmth of the first rays.",
    "The scent of pine filled the air as they walked through the forest.",
    "She opened the book and lost herself in the story.",
    "The waves crashed against the shore, creating a rhythmic sound that lulled him to sleep.",
    "He picked up the old, worn-out guitar and began to play a familiar tune."
]

    


seeds = [random.randint(0,1000000) for _ in range(10)]

with torch.no_grad():
    for seed in seeds:
        generator = torch.manual_seed(seed)

        # embed_list = []
        # for prompt in prompt_list:
        #     embed = simply_encode_prompt(prompt=prompt, text_encoder=text_encoder, tokenizer=tokenizer)
        #     embed_list.append(embed.unsqueeze(0).detach().cpu())
        # all_embed = torch.cat(tuple(embed_list),dim =0)
        # coefs = torch.rand(100)
        # result = torch.einsum('i,ijm->jm',coefs,all_embed)/50

        # seeds = [random.randint(0,1000000) for _ in range(30)]
        # for seed in seeds:
        #     #     aux_prompt_use = True
        #     generator = torch.manual_seed(seed)
        #         # image = pipe(prompt_embeds=ensemble_embeddding, generator = generator).images[0]  
        #     image = pipe(prompt_embeds = result.unsqueeze(0), generator = generator).images[0]  
        #     save_image(image = image, dir = f'./generation_outputs/linear_whole_prompt', file_name = f'{seed}')
            #     # image = pipe(prompt = prompt2, generator = generator).images[0]  
            #     # save_image(image = image, dir = f'./generation_outputs/swap', file_name = f'{seed}_2')
            #     image = pipe(prompt_embeds = mix, generator = generator).images[0]  
            #     save_image(image = image, dir = f'./generation_outputs/swap', file_name = f'{seed}_x')
"""write in file

    with open('./generation_outputs/linear/prompts.txt','a+') as f:
        f.write('\nprompts:\n')
        for p in prompt_list:
            f.write(f'{p}\n')
"""
            
"""random_coef_generation
        embed_list = []
        for prompt in prompt_list:
            embed = simply_encode_prompt(prompt=prompt, text_encoder=text_encoder, tokenizer=tokenizer)
            embed_list.append(embed.unsqueeze(0).detach().cpu())
        all_embed = torch.cat(tuple(embed_list),dim =0)
        coefs = torch.rand(100)
        result = torch.einsum('i,ijm->jm',coefs,all_embed)/50

        seeds = [random.randint(0,1000000) for _ in range(30)]
        for seed in seeds:
            #     aux_prompt_use = True
            generator = torch.manual_seed(seed)
                # image = pipe(prompt_embeds=ensemble_embeddding, generator = generator).images[0]  
            image = pipe(prompt_embeds = result.unsqueeze(0), generator = generator).images[0]  
            save_image(image = image, dir = f'./generation_outputs/linear_whole_prompt', file_name = f'{seed}')


"""