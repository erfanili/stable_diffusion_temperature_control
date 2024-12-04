import json
import os
from collections import defaultdict
import base64
import requests
import time
import pickle as pkl
# prompt_category = 'texture_val'
with open(f'./color_train_question.json', 'r') as f:
    q_dict = json.load(f)

prompt_dir = '/home/erfan/repos/stable_diffusion_temperature_control/prompt_files/comp_bench/json/word_n_index/sd1_5x/color_train.json'
img_dir= '/home/erfan/jy_trial/hyperparam_test/color/images/sd1_5x_2/processor_x_2'

with open(prompt_dir,'r') as f:
    prompt_dict = json.load(f)


def get_the_right_path(directory = './', attr1 = '', obj1 = '', attr2 = '', obj2 = ''):
    
    part1 = ' '.join([attr1,obj1,'and','a',attr2,obj2])
    # part2 = ' '.join([attr2,obj2])
    
    path_list = os.listdir(directory)
    right_paths = []
    for path in path_list:
        if (part1 in path) :
            right_paths.append(path)
            
    return right_paths


paths_dict = {}
for key, value in prompt_dict.items():
    
    adj1 = value[0]['adj1']
    noun1 = value[0]['noun1']
    adj2 = value[0]['adj2']
    noun2 = value[0]['noun2']
    
    paths = get_the_right_path(directory= img_dir,attr1 = adj1, obj1 = noun1, attr2 = adj2, obj2 = noun2)
    
    paths_dict[key] = paths
# print(paths_dict)
# for key,value in paths_dict.items():
#     if len(value) !=3:
#         print(key)
# exit()
    
    
    
    
    
    
def match_image_questions(img_dir, q_dict):
    filenames = os.listdir(img_dir)
    file_dict = defaultdict(list)
    for f in filenames:
        file_dict[f.split('_')[0]].append(f)
    question_image_match_dict = {}
    for q_k in q_dict.keys():
        prompt_id = q_k.split('_')[1]
        question_image_match_dict[q_k] = file_dict[prompt_id]
    return question_image_match_dict

    
question_image_dict = match_image_questions(img_dir, q_dict)





# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image


headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

label_result_dict = {}
failure_cases =[]
for question_key, question in q_dict.items():
    q_id = question_key.split('_')[1]

    print("question key", question_key)

    image_paths = paths_dict[q_id]
    for image_path in image_paths:
        # Getting the base64 string
        image_full_path = os.path.join(img_dir, image_path)
        base64_image = encode_image(image_full_path)

        payload = {
        "model": "gpt-4o-2024-08-06",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                # "text": f"{q_dict['obj_0_1']}"
                "text": f"{question} if yes, print 1, else print 0"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 10
        }

        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            label = response.json()['choices'][0]['message']['content']
            label_result_dict[(question_key, image_path.split('.')[0])] = label
        except:
            print('fail case', image_path)
            failure_cases.append(image_path)
            continue
        #   print("error!")
        #   time.sleep(60)
        #   response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        #   label = response.json()['choices'][0]['message']['content']
        #   label_result_dict[(question_key, image_path.split('.')[0])] = label
        with open(f'color_train_test_labels.pkl', 'wb') as f:
            pkl.dump(label_result_dict, f)

# print("failure :", failure_cases)
