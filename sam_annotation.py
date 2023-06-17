# importing the module
import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import matplotlib.pyplot as plt


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


# loading sam
sam_checkpoint = "D:/Charkh/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

# function to display the coordinates of
# of the points clicked on the image
points = []
labels = []
def click_event(event, x, y, flags, params):
    
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)
        points.append([x, y])
        labels.append(1)
        # print(points)

    # checking for right mouse clicks	
    if event==cv2.EVENT_RBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x,y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', img)
        points.append([x, y])
        labels.append(0)



# driver function
if __name__=="__main__":
    image_path = "./clothes.jpg"
    # reading the image
    img = cv2.imread(image_path, 1)

    # displaying the image
    cv2.imshow("image", img)

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)
    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()

    input_label = np.array(labels)
    input_point = np.array(points)

    
    # print(input_label)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)

    # plt.figure(figsize=(10,10))
    # plt.imshow(image)
    # show_points(input_point, input_label, plt.gca())
    # plt.axis('on')
    # plt.show()  
    masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
    )

    mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

    masks, _, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    mask_input=mask_input[None, :, :],
    multimask_output=False,
    )

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(masks, plt.gca())
    # show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.show() 

    # for i, (mask, score) in enumerate(zip(masks, scores)):
        # plt.figure(figsize=(10,10))
        # plt.imshow(image)
        # show_mask(mask, plt.gca())
        # # show_points(input_point, input_label, plt.gca())
        # plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        # plt.axis('off')
        # plt.show()
