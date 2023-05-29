import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np


model_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
# model_dino = torch.hub.dinov2_vits14()

img = Image.open('/home/art/Code/Auto_seg/dataset/test/aa.jpg')

transform = T.Compose([
T.Resize(256),
T.CenterCrop(224),
T.ToTensor(),
T.Normalize(mean=[0.5], std=[0.5]),
])

img = transform(img)[:3].unsqueeze(0)

with torch.no_grad():
    # features = model_dino.forward_features(img)
    features = model_dino.get_intermediate_layers(img)[0]


print(features)
print(features.shape)


import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

fe = features[0]
fe = (fe - fe.min()) / (fe.max() - fe.min())
fe = fe * 255
plt.imshow(np.array(fe).reshape(256, 256,3).astype(np.uint8))
plt.savefig('features.png')


pca = PCA(n_components=90)
pca.fit(features[0])

pca_features = pca.transform(features[0])
pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
pca_features = pca_features * 255

print(pca_features.shape)

plt.imshow(pca_features.reshape(256, 30, 3).astype(np.uint8))
plt.savefig('features_pca.png')

# def forward(self, *args, is_training=False, return_patches=False, **kwargs):
#     ret = self.forward_features(*args, **kwargs)
#     if is_training:
#         return ret
#     elif return_patches:
#         return ret["x_norm_patchtokens"]
#     else:
#         return self.head(ret["x_norm_clstoken"])