#example command line
#python3 prune.py --view_all_layers False --model_path ../saved_models/unet_resnet18_model_real.pt --amount_to_prune 50

#python3 prune.py --view_all_layers True --model_path ../saved_models/fpn_resnet18_model.pt --amount_to_prune 50
#python3 prune.py --view_all_layers True --model_path ../saved_models/fpn_vgg16_bn_model.pt --amount_to_prune 50
#python3 prune.py --view_all_layers True --model_path ../saved_models/PSPNet_resnet18_model.pt --amount_to_prune 50



#python3 prune.py --view_all_layers True --model_path ../saved_models/FPN_ResNet18_12classes_xingyu.pt --amount_to_prune 50 --view_all_layers True --save False
#segmentation_models.pytorch/saved_models/FPN_ResNet18_12classes_xingyu.pt
#segmentation_models.pytorch/saved_models/FPN_ResNet18_12classes_xingyu.pt
#segmentation_models.pytorch/saved_models/PSPNet_ResNet18_12classes_xingyu.pt
import argparse

parser = argparse.ArgumentParser(description='Prune several different variations.')
parser.add_argument('-amount','--amount_to_prune', type = float, required = False)
parser.add_argument('--model_path', type = str, required = True)
parser.add_argument('--prune_type', type = str, required = False, default = None)
parser.add_argument('--view_all_layers', type = str, required = False, default = 'False')
parser.add_argument('--save_folder', required = False, default = "")
parser.add_argument('--save', required = False, default = "False")
parser.add_argument('--seg_block', required = False, default = "0")

args = parser.parse_args()
#segmentation_models.pytorch/saved_models/unet_resnet18_model_real.pt
import torch
from torch import nn
import torch.nn.utils.prune as prune


#loading base model
#model = torch.load(args.model_path,map_location=torch.device('cpu'))
model = torch.load(args.model_path)

#where the pruning happens
percentage_prune = args.amount_to_prune/100

#see all layers:
if args.view_all_layers == "True":
    for name, module in model.named_modules():
        print(name)
        # if "decoder" in name and "decoder" != name and name != "decoder.psp" and name != "decoder.psp.blocks":
        #     split = name.split(".")
        #     if split[1] == "psp":
        #         if split[3] == args.seg_block:
        #             #print(name)
        #             if isinstance(module,torch.nn.Conv2d):
        #                 print(name,module)


#######################################################################################################################################
#for PSPNet testing 

#prune the conv layers after psp blocks
if args.prune_type == "last_convs":
    for name, module in model.named_modules():
        if "decoder" in name and "decoder" != name:
            split = name.split(".")
            if split[1] == "conv":
                if isinstance(module,torch.nn.Conv2d):
                    prune.l1_unstructured(module, name='weight', amount=percentage_prune)
                    prune.remove(module, 'weight')

if args.prune_type == "psp_block":
    for name, module in model.named_modules():
        if "decoder" in name and "decoder" != name and name != "decoder.psp" and name != "decoder.psp.blocks":
            split = name.split(".")
            if split[1] == "psp":
                if split[3] == args.seg_block:
                    if isinstance(module,torch.nn.Conv2d):
                        prune.l1_unstructured(module, name='weight', amount=percentage_prune)
                        prune.remove(module, 'weight')

if args.prune_type == "not_psp_block":
    for name, module in model.named_modules():
        if "decoder" in name and "decoder" != name and name != "decoder.psp" and name != "decoder.psp.blocks":
            split = name.split(".")
            if split[1] == "psp":
                if split[3] != args.seg_block:
                    if isinstance(module,torch.nn.Conv2d):
                        prune.l1_unstructured(module, name='weight', amount=percentage_prune)
                        prune.remove(module, 'weight')



if args.prune_type == "all_psp_block":
    for name, module in model.named_modules():
        if "decoder" in name and "decoder" != name and name != "decoder.psp" and name != "decoder.psp.blocks":
            split = name.split(".")
            if split[1] == "psp":
                if isinstance(module,torch.nn.Conv2d):
                    prune.l1_unstructured(module, name='weight', amount=percentage_prune)
                    prune.remove(module, 'weight')


#######################################################################################################################################
#for FPN testing
#p2-p5
if args.prune_type == "p2_p5":
    for name, module in model.named_modules():
        if "decoder" in name and "decoder" != name:
            parts = name.split(".")
            #print(parts)
            if parts[1][0] == "p":
                if isinstance(module,torch.nn.Conv2d):
                    prune.l1_unstructured(module, name='weight', amount=percentage_prune)
                    prune.remove(module, 'weight')

#seg blocks individual
if args.prune_type == "seg_block":
    for name, module in model.named_modules():
        if "decoder" in name and "decoder" != name and "decoder.seg_blocks" != name:
            parts = name.split(".")
            if parts[1] == "seg_blocks":
                if parts[2] == args.seg_block:
                    if isinstance(module,torch.nn.Conv2d):
                        prune.l1_unstructured(module, name='weight', amount=percentage_prune)
                        prune.remove(module, 'weight')

#seg blocks uniformly
if args.prune_type == "all_seg_block":
    for name, module in model.named_modules():
        if "decoder" in name and "decoder" != name and "decoder.seg_blocks" != name:
            parts = name.split(".")
            if parts[1] == "seg_blocks":
                if isinstance(module,torch.nn.Conv2d):
                    prune.l1_unstructured(module, name='weight', amount=percentage_prune)
                    prune.remove(module, 'weight')

#seg blocks all but one
if args.prune_type == "not_seg_block":
    for name, module in model.named_modules():
        if "decoder" in name and "decoder" != name and "decoder.seg_blocks" != name:
            parts = name.split(".")
            if parts[1] == "seg_blocks":
                if parts[2] != args.seg_block:
                    if isinstance(module,torch.nn.Conv2d):
                        prune.l1_unstructured(module, name='weight', amount=percentage_prune)
                        prune.remove(module, 'weight')

#######################################################################################################################################
#for resnet model
#encoder
if args.prune_type == "encoder" or args.prune_type == "all_layers":
    for name, module in model.named_modules():
        if "encoder" in name:
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=percentage_prune)
                prune.remove(module, 'weight')

#decoder layers
if args.prune_type == "decoder" or args.prune_type == "all_layers":
    for name, module in model.named_modules():
        if "decoder" in name:
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=percentage_prune)
                prune.remove(module, 'weight')



#for the unet
'''
#for vgg model
#encoder
if args.prune_type == "encoder" or args.prune_type == "all_layers":
    for name, module in model.named_modules():
        if "encoder" in name:
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=percentage_prune)
                prune.remove(module, 'weight')

#decoder layers
if args.prune_type == "decoder" or args.prune_type == "all_layers":
    for name, module in model.named_modules():
        name_split = name.split(".")
        if name_split[0] == 'decoder' and len(name_split) == 5 and name_split[-1] == '0':
                #print(name)
                #print(name,module)
                prune.l1_unstructured(module, name='weight', amount=percentage_prune)
                prune.remove(module, 'weight')
'''

'''
#for resnet model
#decoder layers
if args.prune_type == "decoder" or args.prune_type == "all_layers":
    for name, module in model.named_modules():
        name_split = name.split(".")
        if name_split[0] == 'decoder' and len(name_split) == 5 and name_split[-1] == '0':
                #print(name)
                #print(name,module)
                prune.l1_unstructured(module, name='weight', amount=percentage_prune)
                prune.remove(module, 'weight')

#encoder layers
if args.prune_type == "encoder" or args.prune_type == "all_layers":
    for name, module in model.named_modules():
        name_split = name.split(".")
        if name_split[0] == "encoder" and len(name_split) == 4:
            if name_split[-1][:4] == "conv":
                #print(name)
                #print(name,module)
                prune.l1_unstructured(module, name='weight', amount=percentage_prune)
                prune.remove(module, 'weight')
'''

'''
#bottle_neck
elif args.prune_type == "bottle_neck"    
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


if args.save == "True":
    location = args.save_folder
    name = "/local_prune_" + str(args.prune_type)+ "_" + str(percentage_prune) + ".pt"
    final_save_loc = location + name

    torch.save(model, final_save_loc)
    print('Model saved!')


#torch.save({
#    'the_model':model.state_dict()
#}, final_save_loc)


