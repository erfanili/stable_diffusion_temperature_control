from typing import Dict
import torch
import numpy as np
from scipy import spatial

from utils import load_model

main_prompt = "A woman sits at a wooden desk, her eyes focused on the open book in front of her, while her smart phone lies beside her chair, the soft light from the nearby window casting a gentle glow over the pages."

positive_prompt_list  = [
    "A woman sits at her desk, engrossed in a book, with her smart phone resting on the chair beside her, as sunlight streams through the window.",
    "A woman is seated at a desk, absorbed in her book, while her smart phone is placed on the chair, the light from the window gently illuminating the scene.",
    "At the desk, a woman reads a book, her smart phone on the chair next to her, as the light from the window filters in softly.",
    "A woman is reading a book at a desk, with her smart phone lying on the chair, and the light from the window shining softly on the pages.",
    "A woman is at her desk, focused on a book, with her smart phone nearby on the chair, as light from the window floods the room.",
    "Seated at the desk, a woman reads a book, her smart phone casually placed on the chair, while the window lets in a warm light.",
    "A woman is at a desk, reading a book, her smart phone resting on the chair, as the light from the window creates a calm atmosphere.",
    "The woman at the desk is deeply absorbed in her book, with her smart phone on the chair beside her, while the light from the window casts a warm glow.",
    "A woman reads a book at her desk, her smart phone on the chair, as sunlight pours in through the nearby window.",
    "Sitting at a desk, a woman is immersed in her book, with her smart phone on the chair, bathed in the soft light from the window.",
    "A woman is reading at her desk, her book open and her smart phone resting on the chair, as the window light filters into the room.",
    "The woman at the desk reads a book, her smart phone on the chair beside her, with light from the window brightening the pages.",
    "A woman is focused on a book at her desk, her smart phone on the chair, while the light from the window gently illuminates the scene.",
    "At the desk, a woman reads intently, her smart phone lying on the chair, as the window lets in a soft light.",
    "A woman reads a book at her desk, her smart phone nearby on the chair, with the window light gently filling the room.",
    "Seated at a desk, a woman is reading a book, with her smart phone placed on the chair, as light from the window softly shines.",
    "A woman is seated at her desk, absorbed in her book, while her smart phone rests on the chair beside her, as light from the window streams in.",
    "The woman at the desk is reading a book, her smart phone lying on the chair, while sunlight filters through the window.",
    "A woman is reading a book at the desk, her smart phone on the chair next to her, with the window light softly illuminating the room.",
    "Sitting at the desk, a woman is immersed in her book, her smart phone on the chair, as sunlight streams in through the window."
    "The woman, sitting at a desk, focuses on reading a book, with her smart phone resting on the chair beside her as the window light gently illuminates the room.",
    "At the desk, a woman is absorbed in her book, while her smart phone lies on the chair, with sunlight pouring in through the window.",
    "A woman sits at her desk, reading a book, with her smart phone on the chair, as the light from the window fills the room.",
    "A woman, seated at her desk, is engrossed in a book, while her smart phone rests on the chair and the window light softly brightens the space.",
    "With her smart phone on the chair, a woman reads a book at the desk, as the window lets in a warm, gentle light.",
    "The woman at the desk, reading a book, has her smart phone on the chair, with light streaming in through the window.",
    "A woman is seated at a desk, focused on a book, her smart phone on the chair nearby, while the light from the window gently fills the room.",
    "A woman sits by the window, reading a book at her desk, while her smart phone lies on the chair beside her.",
    "Engrossed in her book, a woman sits at the desk, her smart phone resting on the chair, with the window light softly illuminating the scene.",
    "The woman at the desk is absorbed in her book, with her smart phone on the chair next to her, as sunlight streams through the window.",
    "A woman sits at her desk, her attention fully on the book in front of her, while her smart phone lies on the chair and the window light softly fills the room.",
    "At the desk, a woman reads a book, with her smart phone on the chair beside her, as light from the window illuminates the scene.",
    "Seated at her desk, a woman is focused on a book, her smart phone placed on the chair, while sunlight filters in through the window.",
    "A woman reads a book at her desk, with her smart phone resting on the chair, as the window light gently brightens the room.",
    "A woman sits at her desk, her eyes on the book in front of her, while her smart phone lies on the chair beside her, with light streaming through the window.",
    "The woman, sitting at the desk, reads a book with her smart phone on the chair, as sunlight pours in through the window.",
    "At the desk, a woman reads a book, her smart phone on the chair beside her, while light from the window gently illuminates the pages.",
    "A woman, seated at a desk, is absorbed in her book, while her smart phone rests on the chair and the window light softly fills the room.",
    "Reading her book at the desk, a woman has her smart phone on the chair next to her, with sunlight streaming through the window.",
    "A woman sits at a desk, reading a book, her smart phone on the chair beside her, while light from the window brightens the room."
]

negative_prompt_list = [
    "A man stands by the window, ignoring the book on the desk, while his smart phone is nowhere in sight, and the chair remains empty.",
    "A woman is walking away from the desk, leaving her smart phone and book on the chair, with the window closed and the room dimly lit.",
    "A woman stands at the window, her smart phone in hand, with the book closed on the chair, while the desk remains untouched.",
    "A woman sits outside, far from her desk, scrolling through her smart phone, with no book in sight, as the window is shut behind her.",
    "A man sits at a cluttered desk, but instead of reading a book, he stares blankly out of the window, leaving his smart phone on the chair.",
    "A woman, uninterested in her book, sits in a chair facing away from the desk, while her smart phone is left forgotten on the window sill.",
    "A man stands at his desk, looking at his smart phone, with the book untouched on the chair, as he pulls the curtains over the window.",
    "A woman is lying on the floor, ignoring both the book on the desk and her smart phone on the chair, with the window darkened.",
    "A woman sits by the window, staring out, while her smart phone and book are left behind on the desk and chair.",
    "A woman stands by the window, holding her smart phone, with the book left closed on the desk, and the chair pushed far away.",
    "A man sits at a desk, but instead of reading a book, he watches his smart phone, while the chair remains empty and the window shut.",
    "A woman, facing the closed window, has her back to the desk and book, with her smart phone placed far away on the chair.",
    "A man sits at a desk with no book, his smart phone off and lying on the chair, while he stares out of a closed window.",
    "A woman, uninterested in the book, leaves it on the desk, while she stands at the window with her smart phone, ignoring the empty chair.",
    "A woman is pacing around the room, avoiding the desk, while her smart phone is left off on the chair, and the book is closed by the window.",
    "A man, far from his desk, holds his smart phone and looks out the window, with the book left untouched on the chair.",
    "A woman stands by the window, uninterested in her book, while her smart phone is hidden on the chair, and the desk remains cluttered.",
    "A man sits on the chair, facing away from the window and the desk, his smart phone in hand and the book left unread.",
    "A woman, with her back to the window, sits on the floor, leaving her smart phone and book on the desk and chair.",
    "A man stands by a closed window, ignoring the book on the desk, while his smart phone is turned off on the chair nearby."
    "Ignoring the book on the desk, a woman stands near the window with her smart phone, while the chair is left empty.",
    "A woman, holding her smart phone, stands with her back to the window, leaving the book unopened on the desk and the chair unused.",
    "Far from the desk, a woman sits on the floor, her smart phone in hand, while the book remains on the chair, and the window is closed.",
    "A man stares out of a closed window, leaving the book on the desk and the smart phone on the chair untouched.",
    "A woman paces around the room, avoiding the chair and desk, with her smart phone off and the book closed as the window stays shut.",
    "Standing by the window, a man is absorbed in his smart phone, leaving the book on the chair and the desk cluttered.",
    "The book sits closed on the chair, while a woman stands at the window with her smart phone, ignoring the desk entirely.",
    "A woman, standing by the window, has no interest in the book or the desk, focusing solely on her smart phone and leaving the chair pushed aside.",
    "With her back to the window, a woman holds her smart phone, leaving the book on the chair and ignoring the desk completely.",
    "A man sits on the floor, far from the window and the desk, with his smart phone in hand and the book left untouched on the chair.",
    "The desk remains untouched, with the book closed and the chair empty, while a woman stands by the window with her smart phone.",
    "A woman walks away from the desk, smart phone in hand, while the book and chair remain by the window, untouched.",
    "At the closed window, a man stares blankly, ignoring both the book on the desk and the smart phone on the chair.",
    "Leaving the desk behind, a woman stands at the window, holding her smart phone, while the book lies forgotten on the chair.",
    "A man, with his back to the window, leaves the book on the desk, holding his smart phone and ignoring the chair.",
    "The book lies on the desk, untouched, as a woman focuses on her smart phone by the window, with the chair pushed aside.",
    "A woman, uninterested in the book, walks away from the desk, holding her smart phone and leaving the chair by the window.",
    "By the window, a woman scrolls through her smart phone, leaving the book on the desk and the chair empty.",
    "A man stands at the window, his smart phone in hand, while the book sits unread on the desk and the chair remains unused.",
    "The desk, with its book left untouched, is ignored by a woman who stands by the window, absorbed in her smart phone."
]




text_encoder = load_model(model_to_load='text_encoder',device = 'cuda:4')
tokenizer = load_model(model_to_load='tokenizer',device ='cuda:4')




if __name__ == "__main__":

        
        # with open('./generation_outputs/log.txt','a+') as f:
        #     f.write(f'position: {position}\n')
        #     f.write(f'    positive: {get_distances(position = position, main_prompt=main_prompt, prompt_list=positive_prompt_list)}\n')
        #     f.write(f'    negative: {get_distances(position = position, main_prompt=main_prompt, prompt_list=negative_prompt_list)}\n')
    with open('./generation_outputs/prompts.txt','a+') as f:
        f.write('\nposition_prompts:\n')
        for p in positive_prompt_list:
            f.write(f'{p}\n')
        
        f.write('\nnegative_prompts:\n')
        for p in negative_prompt_list:
            f.write(f'{p}\n')
         