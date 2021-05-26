import torch
from mydata import load_data
import argparse
import os #I'll need to parse through the dataset folder

def dice_coefficient(prediction, ground_truth):
    prediction = prediction.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()
    prediction = np.round(prediction).astype(int)
    ground_truth = np.round(ground_truth).astype(int)
    return np.sum(prediction[ground_truth == 1]) * 2.0 / (np.sum(prediction) + np.sum(ground_truth))


model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)

#testing pruned model

#retreive dictionary of weights

parser = argparse.ArgumentParser(description='Test different pruning variations.')
parser.add_argument('--file_write', required = True)
parser.add_argument('--CHECKPOINT_PATH', required = True)

args = parser.parse_args()


f = open(args.file_write,"a")
f.write(args.CHECKPOINT_PATH)

CHECKPOINT_PATH = args.CHECKPOINT_PATH
dict1 = torch.load(CHECKPOINT_PATH)
model.load_state_dict(dict1['the_model'])

import numpy as np
from PIL import Image
from torchvision import transforms


data_folder = "dataset/kaggle_3m"
folders_list = os.listdir(data_folder)
total_folders = len(folders_list)


absolute_average = 0
non_zero_average = 0
n_abs = 0
n_non = 0

for j in range(total_folders):

    test_images_path = data_folder + "/" + folders_list[j]
    imgs_test, imgs_mask_test = load_data(test_images_path)

    for i in range(imgs_test.shape[0]):
        input_image = imgs_test[i,:,:,:]
        GT = imgs_mask_test[i,:,:,:]

        m, s = np.mean(input_image, axis=(0, 1)), np.std(imgs_test, axis=(0, 1))
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=m, std=s),
        ])
        change = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=m2, std=s2),
        ])
        

        input_tensor = preprocess(input_image)
        output_tensor = change(GT)
        input_batch = input_tensor.unsqueeze(0)


        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model = model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)
        output = torch.round(output[0])

        val = dice_coefficient(output,output_tensor)
        if val>0:
            non_zero_average += val
            n_non+=1
        absolute_average+=val
        n_abs+=1
        #print(f"Calculated val is: {val}")

f.write("\n")
f.write(str(absolute_average/n_abs))
f.write("\n")
f.write(str(n_abs))
f.write("\n")

f.write(str(non_zero_average/n_non))
f.write("\n")
f.write(str(n_non))
f.write("\n\n")
f.close()
print(absolute_average/n_abs)
print(n_abs)
print("\n")
print(non_zero_average/n_non)
print(n_non)
