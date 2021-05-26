# In this code, you have to manually change the values each time. I only learned about shell scripts later.
# run this code first, then you can run test.


import model
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import copy
import numpy as np

# ckpt can be found on SuperSloMo Github
CHECKPOINT_PATH = '/model_path/SuperSloMo.ckpt'
dict1 = torch.load(CHECKPOINT_PATH)

#initialize models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ArbTimeFlowIntrp = model.UNet(20, 5)
ArbTimeFlowIntrp.to(device)
flowComp = model.UNet(6, 4)
flowComp.to(device)

#load weights
ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
flowComp.load_state_dict(dict1['state_dictFC'])

the_model = ArbTimeFlowIntrp

count = 0

#Unstructured based on norm, local
for name, module in the_model.named_modules():
    count += 1

    #this prunes down sample layers
    #if 4<= count<= 18

    #this prunes up sample layers
    #if 19<= count<= 33

    #this prunes only the up sample and down sample layers
    #if 4<= count <=33
 
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.1)
        prune.remove(module, 'weight')
        prune.l1_unstructured(module, name='bias', amount=0.1)
        prune.remove(module, 'bias')



torch.save({
    'state_dictAT':the_model.state_dict(),
    'state_dictFC':flowComp.state_dict()
}, '/save_path/L1_local_unstruct_model_0.1_0.1.ckpt')

