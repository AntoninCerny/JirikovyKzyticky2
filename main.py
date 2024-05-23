#imports for Keypoint RCNN
import torch
import torchvision
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from cjm_pytorch_utils.core import get_torch_device

#imports for picture
from PIL import Image
import numpy as np

#import dataset module
from datasetModule import ClassDataset,tuple_batch
from torch.utils.data import  DataLoader

#import training core module
from coreModule import train_loop
#import plotting module function
from plotPredictedResultModule import plot_prediction


#for checking sqrt(0) or NaN loss etc.
torch.autograd.set_detect_anomaly(True)

#get device
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = get_torch_device()

#START inspired by Pytorch example
#version 1
"""
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

"""



#version 2
anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                   pretrained_backbone=True,
                                                                   num_keypoints=3,
                                                                   num_classes = 2, # Background is the first class, object is the second class
                                                                   rpn_anchor_generator=anchor_generator)


#get dataset

train_dataset = ClassDataset(root="train", transform=None, demo=False)
valid_dataset = ClassDataset(root="valid", transform=None, demo=False)
#STOP inspired by Pytorch example

#START inspired Christian Mills tutorial 
#https://christianjmills.com/posts/pytorch-train-keypoint-rcnn-tutorial/
# Set the training batch size
bs = 1#4

#turn on/off multiprocessing while loading data
num_workers = 0#multiprocessing.cpu_count()

# parameters for DataLoader
data_loader_params = {
    'batch_size': bs,  
    'num_workers': num_workers,  
    'persistent_workers': False,#true  # not using multiprocessing 
    'pin_memory': device,  #'cuda' in # shifts tensor to device memory
    'pin_memory_device': device, #if device == 'cuda'  device else '',  # to GPU
    'collate_fn': tuple_batch,
}

# Dataloader, shuffle helps with overfitting
train_dataloader = DataLoader(train_dataset, **data_loader_params, shuffle=True)

# Dataloader valid
valid_dataloader = DataLoader(valid_dataset, **data_loader_params)
#STOP inspired Christian Mills tutorial 

#setup training loop
checkpoint_dir = "save"
checkpoint_path = checkpoint_dir+"/KeypointRCNN.pth"
epochs = 10


# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
                                params,
                                lr=0.01, #TODO finetune
                                momentum=0.1, 
                                weight_decay=0.0005
                                )  


train_loop(model = model, 
               train_dataloader = train_dataloader, 
               valid_dataloader = valid_dataloader, 
               device = device, 
               epochs = epochs, 
               checkpoint_path = checkpoint_path, 
               optimizer=optimizer,  
               lr_scheduler=None,
               use_scaler=False)



#check model on single image 

#get sample image and make it a tensor
sample_img_path = 'train/images/RgbImage_2022-05-10_10-45-24_png.rf.2b39fa5ae7c670c29d6c2492a26f3a30.jpg'
img = Image.open(sample_img_path)
#img = imgRAW.convert('RGB')
imgWidth, imgHeight  = img.size

image_np = np.array(img)

image_tensor = torch.from_numpy(image_np)
image_tensor = image_tensor.permute(2, 0, 1)
print(image_tensor.shape)
image_tensor = image_tensor.float() / 255.0


image_tensor = image_tensor.to(device)
model.to(device)
#setting model to eval mode
model.eval()
x = [image_tensor]
predictions = model(x)
print(predictions)


plot_prediction(sample_img_path,predictions[0])