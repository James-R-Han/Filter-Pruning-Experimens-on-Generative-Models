# again, this is very manual labour, but it gets better in the Camvid one :)
# you have to comment and uncomment the different pruning modules (but you can easily add it as if-else blocks and add it as a command line arguement

import argparse

parser = argparse.ArgumentParser(description='Prune several different variations.')
parser.add_argument('-amount','--amount_to_prune', type = float, required = True)
parser.add_argument('--save_folder', required = True)

args = parser.parse_args()

import torch
from torch import nn
import torch.nn.utils.prune as prune


#loading base model
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)


#where the pruning happens
percentage_prune = args.amount_to_prune/100

#prune all layers
for name, module in model.named_modules():
    if name == "conv":
        continue
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ConvTranspose2d):
            print(name,module)
            #prune.l1_unstructured(module, name='weight', amount=percentage_prune)
            #prune.remove(module, 'weight')

'''
#bottle_neck
for name, module in model.named_modules():
    stage = name.split(".")
    if len(stage) == 1:
        continue
    temp = stage[0][:3]
    #print(name)
    if temp == "bot":
        if isinstance(module, torch.nn.Conv2d):
            print(name,module)
            #prune.l1_unstructured(module, name='weight', amount=percentage_prune)
            #prune.remove(module, 'weight')
'''

'''
#upsample_convtrapose_and_conv2d
for name, module in model.named_modules():

    if isinstance(module, torch.nn.ConvTranspose2d):
        #print(name,module)
        prune.l1_unstructured(module, name='weight', amount=percentage_prune)
        prune.remove(module, 'weight')
        continue

    stage = name.split(".")
    if len(stage) == 1:
        continue
    temp = stage[0][:3]
    
    if temp == "dec":
        if isinstance(module, torch.nn.Conv2d):
            #print(name,module)
            prune.l1_unstructured(module, name='weight', amount=percentage_prune)
            prune.remove(module, 'weight')
'''


'''
#upsample conv2d
for name, module in model.named_modules():
    stage = name.split(".")
    if len(stage) == 1:
        continue
    temp = stage[0][:3]
    
    if temp == "dec":
        if isinstance(module, torch.nn.Conv2d):
            #print(name,module)
            prune.l1_unstructured(module, name='weight', amount=percentage_prune)
            prune.remove(module, 'weight')
'''

'''
#downconv layers
for name, module in model.named_modules():
    stage = name.split(".")
    if len(stage) == 1:
        continue
    temp = stage[0][:3]
    if temp == "enc":
        if isinstance(module, torch.nn.Conv2d):
            #print(name,module)
            prune.l1_unstructured(module, name='weight', amount=percentage_prune)
            prune.remove(module, 'weight')
'''


'''
#convtranpose2d prune
for name, module in model.named_modules():
    if isinstance(module, torch.nn.ConvTranspose2d):
        #print(name,module)
        prune.l1_unstructured(module, name='weight', amount=percentage_prune)
        prune.remove(module, 'weight')
'''

'''
location = args.save_folder
name = "/local_prune_" + str(percentage_prune) + ".ckpt"
final_save_loc = location + name

torch.save({
    'the_model':model.state_dict()
}, final_save_loc)

'''
