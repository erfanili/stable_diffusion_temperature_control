from utils import *
import torch.nn.functional as F
tokenizer = load_model(model_to_load='tokenizer', device = 'cuda:4')
text_model = load_model(model_to_load='text_model',device='cuda:4')


token = 'dog'
with torch.no_grad():
    inputs = tokenizer(text=token, return_tensors="pt")
    print(inputs['input_ids'])
    embed = text_model(**inputs.to(text_model.device)).last_hidden_state
    embed = embed[:,0,:]
    embedding_matrix = text_model.get_input_embeddings().weight
    # print(embed.size())
    # print(embedding_matrix.size())
    # similarity = F.cosine_similarity(embed, embedding_matrix,dim = -1)
    # print(similarity)
    # predictions = torch.argmax(similarity,dim = -1)
    # print(predictions)
    
    
    
