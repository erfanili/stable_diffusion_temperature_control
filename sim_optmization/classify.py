import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from compute_sim import *
import torch.optim as optim
# 1. Load the dataset from file
class TensorDataset(Dataset):
    def __init__(self, data):
        # data is a list of (tensor, label) tuples
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieves the tensor and label
        ca_map, label, text_sa = self.data[idx]
        return ca_map, label, text_sa
dataset = torch.load('./labeled_maps_dataset.pt')

# text_sa_data = load_pkl(directory='/home/erfan/repos/stable_diffusion_temperature_control/generation_outputs/s_t_optimization/color_train/text_sa/sd1_5x/processor_x',file_name='color_train')

device = 'cuda:0'


# 2. Define Binary Cross-Entropy Loss
ce_loss = nn.CrossEntropyLoss()  # For raw logits, use BCEWithLogitsLoss

# Create a DataLoader for batching
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

model =GetRep(sim_type = 'iou', rep_type ='abs')

optimizer = optim.Adam(model.parameters(), lr=0.01)

model.to(device)


for epoch in range(300):
    epoch_loss = 0
    # print([param for param in model.named_parameters()])
    for batch_idx, (ca_maps, labels,text_sa) in enumerate(dataloader):
        ca_maps = ca_maps.to(device)
        labels = labels.to(device)
        text_sa = text_sa.to(device)

        target_matrix = text_sa[:,1:9,1:9].to(device)

        out = model(ca_maps,target_matrix)

        optimizer.zero_grad()
        loss = ce_loss(out, labels)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print('epoch:',epoch, 'loss:',epoch_loss)
    print(model.conv.bias)
print('complete')
# print([param for param in model.named_parameters()])