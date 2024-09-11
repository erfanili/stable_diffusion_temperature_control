
from utils.load_save_utils import *
from utils.tensor_array_utils import *



get_words = False

if get_words:
    tokenizer = MyTokenizer(model_name = 'sd1_5', device = 'cuda:0')
    txt_directory = './prompt_files/txt/'
    json_words_directory = './prompt_files/json/words/'
    file_names = get_file_names(directory=txt_directory)
    for name in file_names:
        obj_words_in_file = {}
        prompt_list = get_prompt_list_by_line(directory=txt_directory, file_name=name)
        for idx, prompt in enumerate(prompt_list):
            words, indices = get_prompt_words_n_indices(prompt.lower(),tokenizer)
            obj_words_in_file[idx] = {'prompt': prompt.lower(), 'words':words, 'indices':indices}
        save_json(directory=json_words_directory, file_name=name,data=obj_words_in_file)



check_ids = False
if check_ids:
    tokenizer = MyTokenizer(model_name = 'sd1_5', device = 'cuda:0')
    # txt_directory = './prompt_files/txt/'
    json_words_directory = './prompt_files/json/words/'
    file_names = get_file_names(directory=json_words_directory)
    for name in file_names:
        all_prompts_data = load_json(directory=json_words_directory,file_name=name)
        for id, entry in all_prompts_data.items():
            words = []
            prompt = entry['prompt']
            prompt_token_ids = tokenizer.simply_tokenize(text = prompt)
            decoded_words = []
            for idx_list in entry['indices'].values():
                token_list = []
                for idx in idx_list:
                    token_list.append(prompt_token_ids[idx])
                decoded_words.append(tokenizer.decode_a_token_id(token_list=token_list))
            if decoded_words != list(entry['words'].values()):
                print(name, id)
                print('decoded_words: ',decoded_words)
                print(entry['words'].values())


switch_template = False 
if switch_template:
    json_dir1 = './prompt_files/json/words_indices'
    json_dir2 = './prompt_files/json/jy_template'
    file_names = get_file_names(directory=json_dir1)
    for name in file_names:
        all_prompts_data = load_json(directory = json_dir1, file_name = name)
        new_data = {}
        for key,value in all_prompts_data.items():
            new_data[f'obj_{key}_1'] = value['indices']['noun1']
            new_data[f'obj_{key}_2'] = value['indices']['noun2']
        save_json(directory=json_dir2, file_name=name,data=new_data)
            

test = False
if test:
    tokenizer = MyTokenizer(model_name = 'pixart', device = 'cuda:1')
    ids, embeddings = tokenizer.simply_tokenize('a desk')
    print(ids, embeddings)





get_attn_gif = True

if get_attn_gif:
    pkl_name = '0_2813'
    pkl_dir = './generation_outputs/pixart_x/test/attn_dicts'
    output_maps_image_dir = './generation_outputs/pixart_x/test/maps'
    prompt = "an apple on a desk near a pencil"
    data = load_pkl(directory=pkl_dir,file_name=pkl_name)
    tokenizer = MyTokenizer(model_name = 'pixart', device = 'cuda:1')
    token_ids = tokenizer.simply_tokenize(text = 'an')
    prompt_token_ids = [   46,  8947,    30,     3,     9,  4808,  1084,     3,     9, 13966, 1]
    ## word_token_ids:
    # an 46
    # apple 8947
    # desk 4808
    # map 
    # = data['block_13'].shape)
    # idx = 1
    # map = data['block_13']
    
    for idx in range(len(prompt_token_ids)):
        for t in range(len(map)):
            output_dir = os.path.join(output_maps_image_dir,f'{idx}')
            save_attn_by_layer(attn_array=map[t,:,:], token = idx, output_dir=output_dir, file_name=f'{t}')
        
    
        images_to_gif(directory=output_dir, output_path=output_maps_image_dir, file_name = f'{idx}')