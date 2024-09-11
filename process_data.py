
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


switch_template = True 
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
    tokenizer = MyTokenizer(model_name = 'pixart', device = 'cuda:0')
    ids = tokenizer.simply_tokenize('oblong')
    print('length: ', len(ids))
    print('token_ids: ',ids)
    for id in ids:
        print('>',tokenizer.decode_a_token_id([id]))
    print('>',tokenizer.decode_a_token_id(ids))

process_attn_data = False
if process_attn_data:

    model_name = 'pixart_x'
    directory =f'./generation_outputs/{model_name}/try/attn_dicts/'


    ##if loading all files in directory:
    files = get_file_names(directory)

    for file_name in files:
        data = load_dict(directory= directory, file_name=file_name)
        print(data.keys())



    #### if loading a single file:
    file_name = '0_8125'
    data = load_dict(directory=directory, file_name=file_name)
    print(data.keys())
