#imports for Keypoint RCNN
from torchvision.models.detection.keypoint_rcnn import KeypointRCNN
import torch
import torchvision
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator




#imports for picture
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib
import numpy as np
import os
from cjm_pytorch_utils.core import move_data_to_device
import torchvision.transforms as T



#import dataset module
from datasetModule import ClassDataset,tuple_batch
from torch.utils.data import  DataLoader

from cjm_pytorch_utils.core import set_seed, get_torch_device

fig, ax = plt.subplots()
# Load the JPG file - thanks chat GPT
img = Image.open('train/images/RgbImage_2022-05-10_09-05-11-png_2_png.rf.970d4724e63f03b4df10f1a33ab0559f.jpg')
#img = imgRAW.convert('RGB')
imgWidth, imgHeight  = img.size

image_np = np.array(img)

image_tensor = torch.from_numpy(image_np)
image_tensor = image_tensor.permute(2, 0, 1)
print(image_tensor.shape)
image_tensor = image_tensor.float() / 255.0

#get device
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = get_torch_device()
#keypoint RCNN init
backbone = torchvision.models.mobilenet_v2().features
backbone.out_channels = 1280

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))


roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                 output_size=7,
                                                 sampling_ratio=2)

keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                         output_size=14,
                                                         sampling_ratio=2)
 # put the pieces together inside a KeypointRCNN model
model = KeypointRCNN(backbone,
                      num_classes=2,
                      num_keypoints=3,
                      rpn_anchor_generator=anchor_generator,
                      box_roi_pool=roi_pooler,
                      keypoint_roi_pool=keypoint_roi_pooler)



#GETTING DATASET

train_dataset = ClassDataset(root="train", transform=None, demo=False)
valid_dataset = ClassDataset(root="valid", transform=None, demo=False)

#END GETTING DATASET 

#SEND DATA TO DATALOADER

# Set the training batch size
bs = 1#4

# Set the number of worker processes for loading data. This should be the number of CPUs available.
num_workers = 0#multiprocessing.cpu_count()

# Define parameters for DataLoader
data_loader_params = {
    'batch_size': bs,  # Batch size for data loading
    'num_workers': num_workers,  # Number of subprocesses to use for data loading
    'persistent_workers': False,#true  # If True, the data loader will not shutdown the worker processes after a dataset has been consumed once. This allows to maintain the worker dataset instances alive.
    'pin_memory': device,  #'cuda' in # If True, the data loader will copy Tensors into CUDA pinned memory before returning them. Useful when using GPU.
    'pin_memory_device': device, #if device == 'cuda'  device else '',  # Specifies the device where the data should be loaded. Commonly set to use the GPU.
    'collate_fn': tuple_batch,
}

# Create DataLoader for training data. Data is shuffled for every epoch.
train_dataloader = DataLoader(train_dataset, **data_loader_params, shuffle=True)

# Create DataLoader for validation data. Shuffling is not necessary for validation data.
valid_dataloader = DataLoader(valid_dataset, **data_loader_params)


#CREATE TRAIN LOOP



#
# Get only the first image and target
for images, targets in train_dataloader:
    # Since batch_size=1, images and targets will each be a list of one element
    MyImages = images[0]
    MyTargets = targets[0]
  
    # Break after the first batch
    break

"""

# Move the model and data to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
images = [img.to(device) for img in MyImages]
targets = [{k: v.to(device) for k, v in t.items()} for t in MyTargets]
"""
MyImages = MyImages.to(device)
#MyTargets = MyTargets
model.to(device)
#trainingon one picture
model.train()
#trgt = getTarget()
#trgt.to(device)
"""
keypoint_rcnn_targets = [
            {'boxes' : boxes[None], 'labels': labels[None], 'keypoints': keypoints[None]}
            for boxes, labels, keypoints in zip(gt_object_bboxes, gt_labels, gt_keypoints_with_visibility)
        ]
"""
MyImages = [MyImages]
MyTargets = [MyTargets]

model(MyImages,move_data_to_device(MyTargets, device))#move_data_to_device(trgt, device)










image_tensor = image_tensor.to(device)
model.to(device)
#setting model to eval mode
#model.eval()
model.eval()
#x = [torch.rand(3, 300, 400)]
x = [image_tensor]
predictions = model(x)
print(predictions)