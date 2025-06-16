import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from torchvision import models, transforms

# ------- 图像加载 -------
def load_images_and_labels(root_folder, transform):
    images, labels, paths = [], [], []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            class_name = os.path.basename(dirpath)
            path = os.path.join(dirpath, filename)
            try:
                image = Image.open(path).convert("RGB")
                images.append(transform(image))
                labels.append(class_name)
                paths.append(path)
            except:
                continue
    return torch.stack(images), labels, paths

# ------- 标签转数字 -------
def label_to_int_map(labels):
    label_set = sorted(set(labels))
    return {label: i for i, label in enumerate(label_set)}

# ------- 聚类匹配函数 -------
def match_clusters_to_labels(true_labels, cluster_labels):
    matched = np.zeros_like(cluster_labels)
    for cluster in np.unique(cluster_labels):
        mask = cluster_labels == cluster
        majority_label = mode(true_labels[mask], keepdims=False).mode
        matched[mask] = majority_label
    return matched

# ------- 特征提取器（resnet50 去掉分类头）-------
def get_feature_extractor(model):
    return torch.nn.Sequential(*list(model.children())[:-1])

# ------- 主评估函数 -------
def evaluate_model_on_dataset(model_path, dataset_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # 加载图像和标签
    images, label_strs, image_paths = load_images_and_labels(dataset_path, transform)
    if len(images) == 0:
        print(f"[Warning] No images in {dataset_path}")
        return None

    label_map = label_to_int_map(label_strs)
    true_labels = np.array([label_map[lbl] for lbl in label_strs])

    # 加载模型
    model = models.resnet50(pretrained=True)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 2)  # 二分类
    # checkpoint = torch.load(model_path, map_location='cpu')
    # model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    extractor = get_feature_extractor(model)

    with torch.no_grad():
        features = extractor(images).squeeze(-1).squeeze(-1).numpy()

    # 聚类
    n_clusters = len(label_map)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(features)

    pred_labels = match_clusters_to_labels(true_labels, cluster_labels)

    # 每类准确率
    class_acc = {}
    inv_map = {v: k for k, v in label_map.items()}
    for i in range(n_clusters):
        idx = (true_labels == i)
        acc = np.mean(pred_labels[idx] == true_labels[idx])
        class_acc[inv_map[i]] = acc

    return class_acc

# ------- 总控制逻辑 -------
def main():
    model_path = '/home/chence/workspace/shm_detection/freezing/smh_detection_code/checkpoints_new/epo_50_srcNAugAme_lr0.0001_ftlr1e-05_ftep5_20250615_164727/training_best_model_2025-06-15_19-18-48.pth'
    dataset_path = '/home/shared_data/salmonella_detection/AugumentedData/AmericanDataAug'

    # with open(model_info_path, 'r') as f:
    #     lines = f.read().strip().splitlines()

    # for line in lines:
        # model_path, dataset_list_str = line.split(',')
        # dataset_names = dataset_list_str.split('|')
        # print(f"\n Evaluating model: {model_path}")

        # for dataset_name in dataset_names:
        #     dataset_path = os.path.join(dataset_root, dataset_name)
        #     print(f" Dataset: {dataset_name}")
    acc = evaluate_model_on_dataset(model_path, dataset_path)
    if acc is not None:
        for cls, a in acc.items():
            print(f"Class '{cls}' Accuracy: {a:.4f}")
    else:
        print("No valid images.")

if __name__ == '__main__':
    main()
