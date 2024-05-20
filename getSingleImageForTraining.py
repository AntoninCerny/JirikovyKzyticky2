annotations_path = "grain/labels/RgbImage_2022-05-10_10-54-17-png_3_png.rf.50423efb3183be4b483150b59451fb37.txt" #self.root     
        
with open(annotations_path, 'r') as file:
            anotationDataString = file.read().split()
            anotationData = [float(x) for x in anotationDataString]
            #boundry rectangle 
            boundryBoxStartX = (imgWidth * anotationData[1]) - (imgWidth * anotationData[3])/2 #(middle of rectangel) - size/2 
            boundryBoxStartY = (imgHeight * anotationData[3]) + (imgHeight * anotationData[4])/2
            boundryBoxSizeX =  imgWidth * anotationData[3]
            boundryBoxSizeY = imgHeight * anotationData[4]

            startingKeypointX = imgWidth * anotationData[5]
            startingKeypointY = imgHeight * anotationData[6]

            centerKeypointX = imgWidth * anotationData[8]
            centerKeypointY = imgHeight * anotationData[9]

            topKeypointX = imgWidth * anotationData[23]
            topKeypointY = imgHeight * anotationData[24]

annotations = {
    "boxes": [[boundryBoxStartX,boundryBoxStartY-boundryBoxSizeY,boundryBoxStartX+boundryBoxSizeX,boundryBoxStartY+boundryBoxSizeY]],  # Example bounding box
    "labels": [1],  # Example label
    "keypoints": [[[startingKeypointX,startingKeypointY,2],[centerKeypointX,centerKeypointY,2],[topKeypointX,topKeypointY,2]]]  # Example keypoints
}

def prepare_image_and_target(image_path, annotations):
    # Open the image
    image = Image.open(image_path).convert("RGB")
    
    # Define a transformation to convert the image to a tensor
    transform = T.ToTensor()
    image = transform(image)
    
    # Prepare the target dictionary
    target = {}
    target["boxes"] = torch.tensor(annotations["boxes"], dtype=torch.float32)
    target["labels"] = torch.tensor(annotations["labels"], dtype=torch.int64)
    target["keypoints"] = torch.tensor(annotations["keypoints"], dtype=torch.float32)
    
    return image, target
image_path = "grain/images/RgbImage_2022-05-10_10-54-17-png_3_png.rf.50423efb3183be4b483150b59451fb37.jpg"

image, target = prepare_image_and_target(image_path, annotations)


images = [image]
targets = [target]

