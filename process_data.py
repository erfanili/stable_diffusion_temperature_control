
from utils.load_save_utils import *
from utils.tensor_array_utils import *



get_words = True

if get_words:
    tokenizer = MyTokenizer(model_name = 'pixart', device = 'cuda:0')
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



check_ids = True
if check_ids:
    tokenizer = MyTokenizer(model_name = 'pixart', device = 'cuda:0')
    # txt_directory = './prompt_files/txt/'
    json_words_directory = './prompt_files/json/words/'
    file_names = get_file_names(directory=json_words_directory)
    for name in file_names:
        all_prompts_data = load_json(directory=json_words_directory,file_name=name)
        for id, entry in all_prompts_data.items():
            words = []
            prompt = entry['prompt']
            prompt_token_ids = tokenizer.simply_tokenize(text = prompt)
            words = []
            for idx_list in entry['indices'].values():
                token_list = []
                for idx in idx_list:
                    token_list.append(prompt_token_ids[idx])
                words.append(tokenizer.decode_a_token_id(token_list=token_list))
            if words != list(entry['words'].values()):
                print(name, id)
                print(words)
                print(entry['words'])





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
