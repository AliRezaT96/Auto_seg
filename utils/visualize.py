import cv2
from pycocotools import coco
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random

# Path to the Fashionpedia annotation file
annotation_file = '/home/art/Code/Auto_seg/dataset/instances_attributes_train2020.json'

# Initialize COCO dataset object
coco_dataset = coco.COCO(annotation_file)

# Load all image IDs in the dataset
image_ids = coco_dataset.getImgIds()

# Iterate over the image IDs
for image_id in image_ids:
    image_id = 10
    # Load image information and annotations for the current image ID
    image_info = coco_dataset.loadImgs(image_id)[0]
    annotations_ids = coco_dataset.getAnnIds(imgIds=image_id)
    annotations = coco_dataset.loadAnns(annotations_ids)

    # Load and display the image
    image_path = '/home/art/Code/Auto_seg/dataset/train2020/train/' + image_info['file_name']
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image)
    plt.imshow(image)
    plt.axis('off')

    color_map = {}

    # Iterate over the annotations and draw the segmentation masks with unique colors
    for annotation in annotations:
        segment_id = annotation['id']
        segmentation = annotation['segmentation']

        # Generate a random color for the segment if not already assigned
        if segment_id not in color_map:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            color_map[segment_id] = color
        else:
            color = color_map[segment_id]

        for seg in segmentation:
            # Reshape the segmentation coordinates
            seg = np.array(seg).reshape((int(len(seg) / 2), 2)).astype(np.int32)

            # Draw the segmentation mask on the image with the assigned color
            cv2.polylines(image, [seg], isClosed=True, color=color, thickness=2)

    # Show the image with segmentation masks
    plt.imshow(image)
    plt.show()


    break
    
