import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

#START inspired by Alex P from Medium but completely redone acording to Pytorch documentation
#https://medium.com/@alexppppp/how-to-train-a-custom-keypoint-detection-model-with-pytorch-d9af90e111da

class ClassDataset(Dataset):
   def __init__(self, root, transform=None, demo=False):              
        self.root = root
        self.transform = transform
        self.demo = demo # Use demo=True if you need transformed and original images (for example, for visualization purposes)
        self.imgs_files = sorted(os.listdir(os.path.join(root, "images")))
        self.annotations_files = sorted(os.listdir(os.path.join(root, "labels"))) #os.path.join(root, "labels")
    
   def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs_files[idx])
        annotations_path = os.path.join(self.root, "labels", self.annotations_files[idx]) #self.root
        

        img_original = Image.open(img_path)
        #img_original = mpimg.imread(img_path)
        imgHeight, imgWidth = img_original.size

        image_np = np.array(img_original)
        image_tensor = torch.from_numpy(image_np)
        image_tensor = image_tensor.permute(2, 0, 1)
        #print(image_tensor.shape)
        image_tensor = image_tensor.float() / 255.0
        image_tensor.to('cuda')
        #img_original = mpimg.cvtColor(img_original, mpimg.COLOR_BGR2RGB)        
        
        with open(annotations_path, 'r') as file:
            anotationDataString = file.read().split()
            anotationData = [float(x) for x in anotationDataString]
            #boundry rectangle 
            boundryBoxStartX = (imgWidth * anotationData[1]) - (imgWidth * anotationData[3])/2 #(middle of rectangel) - size/2 
            boundryBoxStartY = (imgHeight * anotationData[3]) + (imgHeight * anotationData[4])/2
            boundryBoxSizeX =  imgWidth * anotationData[3]
            boundryBoxSizeY = imgHeight * anotationData[4]

            bboxes_original =[boundryBoxStartX,boundryBoxStartY-boundryBoxSizeY,boundryBoxStartX+boundryBoxSizeX,boundryBoxStartY+boundryBoxSizeY]
            bboxes_original = np.array(bboxes_original) #convert to numpy array 


            startingKeypointX = imgWidth * anotationData[5]
            startingKeypointY = imgHeight * anotationData[6]

            centerKeypointX = imgWidth * anotationData[8]
            centerKeypointY = imgHeight * anotationData[9]

            topKeypointX = imgWidth * anotationData[11]
            topKeypointY = imgHeight * anotationData[12]

            keypoints_original = [[[startingKeypointX,startingKeypointY,1],[centerKeypointX,centerKeypointY,1],[topKeypointX,topKeypointY,1]]]

            # All objects are glue tubes
        #bboxes_labels_original = ['Glue tube' for _ in bboxes_original]            

        bboxes, keypoints = bboxes_original, keypoints_original        
        
        # Convert everything into a torch tensor        
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        #changing size of tensor for [4] to [1,4] for compatibility with the model 
        bboxes = bboxes.unsqueeze(0)  
        keypoints_tensor = torch.as_tensor(keypoints, dtype=torch.float32) 
        #print(keypoints_tensor.shape)
        #print(keypoints_tensor)

        annotations = {
            "boxes": [[boundryBoxStartX,boundryBoxStartY-boundryBoxSizeY,boundryBoxStartX+boundryBoxSizeX,boundryBoxStartY+boundryBoxSizeY]],  # Example bounding box
            "labels": [1],  # Example label
            "keypoints": [[[startingKeypointX,startingKeypointY,2],[centerKeypointX,centerKeypointY,2],[topKeypointX,topKeypointY,2]]]  # Example keypoints
        }


        # Prepare the target dictionary
        target = {}
        target["boxes"] = torch.tensor(annotations["boxes"], dtype=torch.float32)
        target["labels"] = torch.tensor(annotations["labels"], dtype=torch.int64)
        target["keypoints"] = torch.tensor(annotations["keypoints"], dtype=torch.float32)


        return image_tensor, target
    
   def __len__(self):
        return len(self.imgs_files)
def tuple_batch(batch):
    return tuple(zip(*batch))


#STOP inspired by Alex P from Medium but completely redone acording to Pytorch documentation