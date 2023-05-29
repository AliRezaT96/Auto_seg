
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load a pre-trained model
model = models.resnet50(pretrained=True)
# Specify the layer you want to extract features from
target_layer = model.layer4[-1]

# Load and preprocess the first image
image1_path = './dataset/test/a.jpg'
image1 = Image.open(image1_path).convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_image1 = preprocess(image1).unsqueeze(0)

# Load and preprocess the second image
image2_path = './dataset/test/aa.jpg'
image2 = Image.open(image2_path).convert('RGB')
input_image2 = preprocess(image2).unsqueeze(0)

# Put the model in evaluation mode
model.eval()

# Extract features from the first image
with torch.no_grad():
    output1 = model(input_image1)
features1 = output1.squeeze()

print(features1.shape)

# Extract features from the second image
with torch.no_grad():
    output2 = model(input_image2)
features2 = output2.squeeze()

print(features2.shape)

# Compute similarity between the two sets of features
similarity = torch.cosine_similarity(features1, features2, dim=0)

# Reshape the similarity tensor to match the dimensions of the second image
similarity_map = similarity.view(image2.size[1], image2.size[0])

# Normalize the similarity map to the range [0, 1]
similarity_map = (similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min())

# Create a heatmap visualization of the similarity map
heatmap = plt.cm.jet(similarity_map)
heatmap = heatmap[:, :, :3]

# Overlay the heatmap onto the second image
blended_image = (heatmap * 0.5 + np.array(image2)) / 1.5

# Plot the second image and the blended image
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image2)
ax[0].set_title('Second Image')
ax[0].axis('off')
ax[1].imshow(blended_image)
ax[1].set_title('Image with Mapped Features')
ax[1].axis('off')
plt.tight_layout()
plt.show()# import torch






# import torchvision.models as models
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image

# # Load a pre-trained model
# # model = models.resnet50(pretrained=True)
# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
# # Specify the layer you want to visualize
# # target_layer = model.layer4[-1]

# # Load and preprocess the input image
# image_path = '/home/art/Code/Auto_seg/dataset/test/a.jpg'
# image = Image.open(image_path).convert('RGB')

# preprocess = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# input_image = preprocess(image).unsqueeze(0)

# print(input_image.shape)
# # Put the model in evaluation mode
# model.eval()

# # Perform a forward pass
# # output = model(input_image)
# output = model.get_intermediate_layers(input_image)[0]

# # Retrieve the desired feature maps
# feature_maps = output.detach().squeeze()

# print(feature_maps.shape)

# # Convert the feature maps to a PIL Image
# pil_feature_maps = transforms.ToPILImage()(feature_maps)

# # Resize the feature maps to match the original image size
# resized_feature_maps = transforms.functional.resize(
#     pil_feature_maps,
#     (image.size[1], image.size[0]),
#     interpolation=Image.BILINEAR
# )



# # Convert the resized feature maps back to a tensor
# resized_feature_maps = transforms.ToTensor()(resized_feature_maps)
# print(resized_feature_maps.shape)
# # Normalize the feature maps
# normalized_feature_maps = torch.nn.functional.normalize(resized_feature_maps, dim=0)

# # Convert the feature maps to a numpy array
# normalized_feature_maps_np = normalized_feature_maps.numpy()

# # Create a heatmap visualization of the feature maps
# heatmap = np.mean(normalized_feature_maps_np, axis=0)
# heatmap = np.clip(heatmap, 0, 1)

# # Overlay the heatmap onto the original image
# heatmap = plt.cm.jet(heatmap)
# heatmap = heatmap[:, :, :3]
# blended_image = (heatmap * 0.5 + np.array(image)) / 1.5
# # blended_image = heatmap
# # Plot the original image and the blended image
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(image)
# ax[0].set_title('Original Image')
# ax[0].axis('off')
# ax[1].imshow(blended_image)
# ax[1].set_title('Image with Features')
# ax[1].axis('off')
# plt.tight_layout()
# plt.show()




# import torch
# from PIL import Image
# import torchvision.transforms as T
# import numpy as np
# import matplotlib.pyplot as plt


# model_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
# # model_dino = torch.hub.dinov2_vits14()

# img = Image.open('/home/art/Code/Auto_seg/dataset/test/aa.jpg')

# transform = T.Compose([
# T.Resize((224,224)),
# T.CenterCrop(224),
# T.ToTensor(),
# T.Normalize(mean=[0.5], std=[0.5]),
# ])

# img = transform(img)[:3].unsqueeze(0)
# print(img.shape)

# with torch.no_grad():
#     # features = model_dino.forward_features(img)
#     features = model_dino.get_intermediate_layers(img)[0]


# # print(features)
# print(features.shape)


# # # Upsample the feature maps
# # upsampled_feature_maps = torch.nn.functional.interpolate(
# #     features[0], size=(256, 256), mode='bilinear', align_corners=False
# # )

# # Upsample the feature maps
# upsampled_feature_maps = torch.nn.functional.interpolate(
#     features.unsqueeze(0),
#     size=(224, 224),
#     mode='bilinear',
#     align_corners=False
# ).squeeze()

# print(upsampled_feature_maps.shape)

# # Normalize the feature maps
# normalized_feature_maps = torch.nn.functional.normalize(upsampled_feature_maps, dim=0)

# # Convert the feature maps to a numpy array
# normalized_feature_maps_np = normalized_feature_maps.numpy()




# # # Create a heatmap visualization of the feature maps
# heatmap = np.mean(normalized_feature_maps_np, axis=0)
# heatmap = np.clip(heatmap, 0, 1)
# print(heatmap.shape)


# # Overlay the heatmap onto the original image
# heatmap = plt.cm.jet(heatmap)
# heatmap = heatmap[:, :, :3]
# blended_image = (heatmap * 0.5 + img) / 1.5

# # Plot the original image and the blended image
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(image)
# ax[0].set_title('Original Image')
# ax[0].axis('off')
# ax[1].imshow(blended_image)
# ax[1].set_title('Image with Features')
# ax[1].axis('off')
# plt.tight_layout()
# plt.show()


# from sklearn.decomposition import PCA

# fe = features[0]
# fe = (fe - fe.min()) / (fe.max() - fe.min())
# fe = fe * 255
# plt.imshow(np.array(fe).reshape(256, 256,3).astype(np.uint8))
# plt.savefig('features.png')


# pca = PCA(n_components=90)
# pca.fit(features[0])

# pca_features = pca.transform(features[0])
# pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
# pca_features = pca_features * 255

# print(pca_features.shape)

# plt.imshow(pca_features.reshape(256, 30, 3).astype(np.uint8))
# plt.savefig('features_pca.png')

# # def forward(self, *args, is_training=False, return_patches=False, **kwargs):
# #     ret = self.forward_features(*args, **kwargs)
# #     if is_training:
# #         return ret
# #     elif return_patches:
# #         return ret["x_norm_patchtokens"]
# #     else:
# #         return self.head(ret["x_norm_clstoken"])