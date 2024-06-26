import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import os

fig, ax = plt.subplots()
# Load the JPG file - thanks chat GPT
img = mpimg.imread('train/images/RgbImage_2022-05-10_10-45-24_png.rf.2b39fa5ae7c670c29d6c2492a26f3a30.jpg')
imgHeight, imgWidth, _ = img.shape

# Load txt file - thanks chat GPT
with open('train/labels/RgbImage_2022-05-10_10-45-24_png.rf.2b39fa5ae7c670c29d6c2492a26f3a30.txt', 'r') as file:
    anotationDataString = file.read().split()
anotationData = [float(x) for x in anotationDataString]


#boundry rectangle 
boundryBoxStartX = (imgWidth * anotationData[1]) - (imgWidth * anotationData[3])/2 #(middle of rectangel) - size/2 
boundryBoxStartY = (imgHeight * anotationData[2]) - (imgHeight * anotationData[4])/2
boundryBoxSizeX =  imgWidth * anotationData[3]
boundryBoxSizeY = imgHeight * anotationData[4]

rectangle = patches.Rectangle((boundryBoxStartX,boundryBoxStartY), boundryBoxSizeX, boundryBoxSizeY, edgecolor='r', facecolor='none')



#keypoints

#top of the tree
if anotationData[7] == 2:
    startingKeypointX = imgWidth * anotationData[5]
    startingKeypointY = imgHeight * anotationData[6]

    startingKeypoint = plt.scatter(startingKeypointX, startingKeypointY, color='red',s = 3)
#center of the tree
if anotationData[10] == 2:
    startingKeypointX = imgWidth * anotationData[8]
    startingKeypointY = imgHeight * anotationData[9]

    startingKeypoint = plt.scatter(startingKeypointX, startingKeypointY, color='red',s = 3)
#bottom of the tree
if anotationData[13] == 2:
    startingKeypointX = imgWidth * anotationData[11]
    startingKeypointY = imgHeight * anotationData[12]

    startingKeypoint = plt.scatter(startingKeypointX, startingKeypointY, color='red',s = 3)


# Display the image using Matplotlib
plt.imshow(img)
ax.add_patch(rectangle)
plt.show()

def plot_target_image(imageName,trainOrValid):
    if trainOrValid == 'train':
        root = "train"
    elif trainOrValid == 'valid':
        root = "valid"
    else:
        raise ValueError("This is the error message")

    
    os.path.join(root, imageName)
    img = mpimg.imread(os.path.join(root, imageName))


    plt.show()





class VisualizeExampleFromDataset():
    """
    This class shows example from Roboflow dataset 
    

    """
