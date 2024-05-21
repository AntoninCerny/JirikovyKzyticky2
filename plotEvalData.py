from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator

from cjm_pytorch_utils.core import get_torch_device
#import plotting module function
from plotPredictedResultModule import plot_prediction



device = get_torch_device()

anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                   pretrained_backbone=True,
                                                                   num_keypoints=3,
                                                                   num_classes = 2, # Background is the first class, object is the second class
                                                                   rpn_anchor_generator=anchor_generator)


# Load the saved parameters into the model
model.load_state_dict(torch.load("save/keypointRCNN.pth"))


sample_img_path = 'train/images/RgbImage_2022-05-10_09-05-11-png_2_png.rf.970d4724e63f03b4df10f1a33ab0559f.jpg'
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
#model.eval()
model.eval()
#x = [torch.rand(3, 300, 400)]
x = [image_tensor]
predictions = model(x)



plot_prediction(sample_img_path,predictions[0])


print(predictions)