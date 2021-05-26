#example to run: python3 test.py --model_path ../saved_models/unet_resnet18_model_real.pt --save_path test.txt
#python3 test.py --model_path ../saved_models/fpn_vgg16_bn_model.pt --save_path test.txt
#python3 test.py --model_path ../saved_models/fpn_resnet18_model.pt --save_path test.txt
#python3 test.py --model_path ../saved_models/fpn_resnet18_model_nearest_interpolation.pt --save_path test.txt
#python3 test.py --model_path ../saved_models/fpn_resnet18_model_changed_order_interpolation.pt --save_path test.txt
#python3 test.py --model_path ../saved_models/Linknet_resnet18_model.pt --save_path test.txt
#required arguements for bash script

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=False)

args = parser.parse_args()


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt


DATA_DIR = '../data/CamVid/'

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
               'tree', 'signsymbol', 'fence', 'car', 
               'pedestrian', 'bicyclist', 'unlabelled']
    
    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None,):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)


###########################################################################################################################################
#Augmentation

import albumentations as albu

def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [albu.CLAHE(p=1),albu.RandomBrightness(p=1),albu.RandomGamma(p=1),],p=0.9,),

        albu.OneOf(
            [albu.IAASharpen(p=1),albu.Blur(blur_limit=3, p=1),albu.MotionBlur(blur_limit=3, p=1),],p=0.9,),

        albu.OneOf(
            [albu.RandomContrast(p=1),albu.HueSaturationValue(p=1),],p=0.9,),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

###################################################################################################################################################

#Testing

import torch
import numpy as np
import segmentation_models_pytorch as smp

#ENCODER = 'resnet18'
ENCODER = 'vgg16_bn'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'
#CLASSES = ['car']
CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol', 'fence', 'car', 'pedestrian', 'bicyclist', 'unlabelled']
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]


# load best saved checkpoint
best_model = torch.load(args.model_path)
#'../saved_models/unet_vgg16_bn_model.pt'

# create test dataset
test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

test_dataloader = DataLoader(test_dataset)


# evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(model=best_model,loss=loss,metrics=metrics,device=DEVICE,)

#unet_resnet18_model_real.pt:  00:07<00:00, 29.74it/s, dice_loss - 0.3078, iou_score - 0.7713
#unet_vgg16_bn_model.plt: 00:18<00:00, 12.66it/s, dice_loss - 0.3052, iou_score - 0.7774

logs = test_epoch.run(test_dataloader)


#save results
with open(args.save_path,"a") as f:
    f.write(f"model: {args.model_path}\n")
    f.write(f"iou: {logs['iou_score']}\n")
    f.write(f"dice_loss: {logs['dice_loss']}\n")

