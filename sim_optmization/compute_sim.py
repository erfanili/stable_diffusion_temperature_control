import sys
import os
# Get the parent directory path and add it to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils.load_save_utils import *
import os
import torch
import torch.nn as nn
 

class GetRep(nn.Module):
    def __init__(self,sim_type = 'iou', rep_type ='abs'):
        super(GetRep,self).__init__()


        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=128,out_features=3)
        # self.fc2 = nn.Linear(in_features=100,out_features=1)
        self.conv = nn.Conv1d(in_channels = 8, out_channels = 128,kernel_size=1,groups=8)
        # self.conv2 = nn.Conv1d(in_channels = 128, out_channels = 8,kernel_size=1,groups=1)
        self.conv.weight = nn.Parameter(torch.ones(8,1,1)*5,requires_grad=False)
        self.conv.bias = nn.Parameter(torch.ones((8,))/2)
        # self.fc_w = nn.Parameter(torch.ones(1))
        # self.fc_b = nn.Parameter(torch.zeros(1))
        self.w1 = nn.Parameter(torch.rand(200))
        # self.fc1 = nn.Linear(in_features=1,out_features=20)
        # self.fc2 = nn.Linear(in_features=20,out_features=20)
        # self.fc3 = nn.Linear(in_features=20, out_features=3)
        self.bn2 = nn.BatchNorm1d(20)
        self.cos2 = nn.CosineSimilarity(dim=1, eps = 1e-6)
        self.sim_type = sim_type
        self.rep_type = rep_type
        self.bn = nn.BatchNorm1d(1,affine = False)
    def iou(self,batched_map1,batched_map2):
        query_length  = batched_map1.shape[1]
        # exit()
        intersection = torch.einsum('ij,ij->i',batched_map1*5,batched_map2*5)
 
        
        union = query_length-torch.einsum('ij,ij->i',1-batched_map1*5,1-batched_map2*5)

        return intersection/union

    def cos1(self,map1,map2):
        query_length = map1.shape[1]
        out = torch.einsum('ij,ij->i',map1,map2)*query_length/(torch.sum(map1)*torch.sum(map2))
        
        return out

    def compute_sim(self,bin_maps):

        batch_size,seq_length, _ =bin_maps.size()
  


        
        batched_sim = torch.zeros(batch_size,seq_length,seq_length).to(bin_maps.device)
        for i in range(seq_length):
            sim_row = torch.zeros(batch_size)
            for j in range(i+1):
                if self.sim_type == 'iou':
                    batched_sim[:,i,j] = self.iou(batched_map1=bin_maps[:,i],batched_map2 = bin_maps[:,j])
                elif self.sim_type == 'cos2':
                    batched_sim[:,i,j] = self.cos2(bin_maps[:,i],bin_maps[:,j])
                elif self.sim_type == 'cos1':
                    batched_sim[:,i,j] = self.cos1(bin_maps[:,i],bin_maps[:,j])
                elif self.sim_type == 'attn_like':
                    batched_sim[:,i,j] = self.cos1(bin_maps[:,i],bin_maps[:,j])
                    sim_row += batched_sim[:,i,j]
                
            if self.sim_type == 'attn_like':
                
                batched_sim[:,i] /= sim_row 
                    
        return batched_sim
    
    
    def get_s_dot_t(self,sim_matrix,target_matrix):
        target_matrix *= 100
        batch_size,seq_length,_ = sim_matrix.size()
        out = torch.zeros(batch_size)
        out = out.to(sim_matrix.device)

        for i in range(seq_length):
            if self.rep_type == 'abs':
                out += torch.sum(torch.abs(sim_matrix[:,i,:i+1]-target_matrix[:,i,:i+1]),dim=1)
            elif self.rep_type == 'cos1':
                out += self.cos1(sim_matrix[:,i,:i+1],target_matrix[:,i,:i+1],dim =1)
            elif self.rep_type == 'cos2':
                out += self.cos2(sim_matrix[:,i,:i+1],target_matrix[:,i,:i+1])
                
        return out

            
    def forward(self,attn_maps,target_matrix):

        attn_maps = (attn_maps - torch.mean(attn_maps.mean(dim=2,keepdim=True)))*100

        bin_maps = self.sigmoid(self.conv(attn_maps))/5
   
        sim = self.compute_sim(bin_maps = bin_maps,
                    )
        st_distance = self.get_s_dot_t(sim,target_matrix)

        st_distance = self.bn(st_distance.unsqueeze(1))

        st_distance = self.fc1(st_distance)
        return st_distance
     



                
    
# attn_maps = torch.rand(size=(8,256))
# target_matrix = torch.rand(size = (8,8))
# threshold_vector = torch.rand(size=(8,))

# rep = get_rep(attn_maps = attn_maps,
#         target_matrix=target_matrix,
#         threshold_vecor=threshold_vector,
#         sim_type='attn_like',
#         rep_type='cos1')

# print(rep)