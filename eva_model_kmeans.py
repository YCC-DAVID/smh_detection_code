import os
import torch
import argparse
import numpy as np
import torch.nn as nn
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from torchvision import models, transforms

parser = argparse.ArgumentParser(description='evaluate model on dataset with clustering')

parser.add_argument('--dataset_path', type=str, required=True,help='Source dataset path')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
parser.add_argument("-pretrain", action='store_true')

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
def evaluate_model_on_dataset(model_path, dataset_path,pretrained=False):
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

    # 加载模型（这里使用 ImageNet 上预训练的 resnet50 提取特征）
    if pretrained:
        model = models.resnet50(pretrained=True)
    else:
        model = models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # 二分类
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    extractor = get_feature_extractor(model)

    with torch.no_grad():
        features = extractor(images).squeeze(-1).squeeze(-1).numpy()

    # 聚类
    n_clusters = len(label_map)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(features)

    # 打印每类数据在各聚类下的分布数量
    inv_map = {v: k for k, v in label_map.items()}
    cluster_distribution = {inv_map[i]: {cid: 0 for cid in range(n_clusters)} for i in range(n_clusters)}

    for t_lbl, c_lbl in zip(true_labels, cluster_labels):
        class_name = inv_map[t_lbl]
        cluster_distribution[class_name][c_lbl] += 1

    print("\n=== Cluster Distribution by True Class ===")
    for class_name, clusters in cluster_distribution.items():
        print(f"Class '{class_name}':")
        for cluster_id, count in clusters.items():
            print(f"  Cluster {cluster_id}: {count} images")

    # 计算每类准确率
    pred_labels = match_clusters_to_labels(true_labels, cluster_labels)
    class_acc = {}
    for i in range(n_clusters):
        idx = (true_labels == i)
        acc = np.mean(pred_labels[idx] == true_labels[idx])
        class_acc[inv_map[i]] = acc

    return class_acc

# ------- 总控制逻辑 -------
def main():
    args = parser.parse_args()
    model_path = args.model_path
    dataset_path = args.dataset_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}") 
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if args.pretrain:
        print("Using pretrained model as baseline.")
    # model_path = '/home/chence/workspace/shm_detection/freezing/smh_detection_code/checkpoints_new/epo_50_srcNAugAme_lr0.0001_ftlr1e-05_ftep5_20250615_164727/training_best_model_2025-06-15_19-18-48.pth'
    # dataset_path = '/home/shared_data/salmonella_detection/AugmentedData/AmericanDataAug'

    # with open(model_info_path, 'r') as f:
    #     lines = f.read().strip().splitlines()

    # for line in lines:
        # model_path, dataset_list_str = line.split(',')
        # dataset_names = dataset_list_str.split('|')
        # print(f"\n Evaluating model: {model_path}")

        # for dataset_name in dataset_names:
        #     dataset_path = os.path.join(dataset_root, dataset_name)
        #     print(f" Dataset: {dataset_name}")
    acc = evaluate_model_on_dataset(model_path, dataset_path,pretrained=args.pretrain)
    print("\n=== Evaluation Results ===")
    if acc is not None:
        for cls, a in acc.items():
            print(f"Class '{cls}' Accuracy: {a:.4f}")
    else:
        print("No valid images.")

if __name__ == '__main__':
    main()
