from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def plot_prediction(image_path, prediction):
    """
    
    
    
    
    """
    # Load the image
    image = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(12, 9))

    # Display the image
    ax.imshow(image)

    # Get the bounding boxes and keypoints from the prediction 
    boxes = prediction['boxes'].to('cpu').detach().numpy()
    keypoints = prediction['keypoints'].to('cpu').detach().numpy()
    boxes = [boxes[1]]
    keypoints = [keypoints[1]]
    #predicted_keypoints = (prediction['keypoints'][0])[:,:,:-1].reshape(-1,2)*1
    # Loop over each detected instance
    for box, kpts in zip(boxes, keypoints):
        # Draw the bounding box
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Draw the keypoints
        for i, (x, y,_) in enumerate(kpts):
            ax.plot(x, y, 'bo')  # Blue dot for keypoints
            ax.text(x, y, f'{i}', color='yellow', fontsize=12)  # Optional: Label the keypoints

    #plt.axis('off')
    plt.show()
