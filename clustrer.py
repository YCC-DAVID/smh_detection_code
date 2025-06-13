import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torchvision import models, transforms
import torch

# ----------- Step 1: 加载图像并提取特征 -----------

def load_images_from_folder(folder, transform, max_images=None):
    images = []
    paths = []
    for i, filename in enumerate(os.listdir(folder)):
        if max_images and i >= max_images:
            break
        path = os.path.join(folder, filename)
        try:
            image = Image.open(path).convert("RGB")
            images.append(transform(image))
            paths.append(path)
        except:
            continue
    return torch.stack(images), paths

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 假设你有一个图像文件夹
image_folder = "path/to/your/images"
image_tensor, image_paths = load_images_from_folder(image_folder, transform, max_images=100)

# 使用预训练的 ResNet18 提取特征（去掉分类头）
resnet = models.resnet18(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.eval()

with torch.no_grad():
    features = feature_extractor(image_tensor).squeeze(-1).squeeze(-1)  # shape [N, 512]
    features_np = features.numpy()

# ----------- Step 2: 特征降维（可选） -----------
pca = PCA(n_components=50)
features_reduced = pca.fit_transform(features_np)

# ----------- Step 3: 聚类分类 -----------

n_clusters = 5  # 你想要的类别数
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
cluster_labels = kmeans.fit_predict(features_reduced)

# ----------- Step 4: 打印结果 -----------

for path, label in zip(image_paths, cluster_labels):
    print(f"Image: {os.path.basename(path)} -> Cluster: {label}")
