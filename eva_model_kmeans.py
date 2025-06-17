import os
import re
import torch
import argparse
from datetime import datetime
import numpy as np
import torch.nn as nn
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import mode
from torchvision import models, transforms

parser = argparse.ArgumentParser(description='evaluate model on dataset with clustering')

parser.add_argument('--dataset_path', type=str, required = True,help='Source dataset path')
parser.add_argument('--model_path', type=str, default = None, help='Path to the model checkpoint')
parser.add_argument("-pretrain", action='store_true')


def list_experiments_with_models(root_dir="checkpoints_new"):
    """
    返回包含至少一个 training_best_model_*.pth 文件的实验目录名列表。
    """
    pattern = re.compile(r'training_best_model_(\d{4}_\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.pth')
    valid_experiments = []

    for name in os.listdir(root_dir):
        exp_dir = os.path.join(root_dir, name)
        if not os.path.isdir(exp_dir):
            continue

        # 检查是否含有匹配模型文件
        has_model = any(pattern.match(f) for f in os.listdir(exp_dir))
        if has_model:
            valid_experiments.append(name)

    return valid_experiments


# 2. 获取某个实验中时间戳最新的模型路径
def find_latest_model_in_experiment(exp_dir):
    """
    给定一个实验路径，返回该实验中时间最新的 best_model 的路径。
    适用于命名格式：training_best_model_2025_06-16_21-19-59.pth
    """
    pattern = re.compile(r'training_best_model_(\d{4}_\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.pth')
    best_models = []

    for filename in os.listdir(exp_dir):
        match = pattern.match(filename)
        if match:
            time_str = match.group(1)
            try:
                time = datetime.strptime(time_str, "%Y_%m-%d_%H-%M-%S")
                full_path = os.path.join(exp_dir, filename)
                best_models.append((time, full_path))
            except ValueError:
                continue

    if not best_models:
        return None
    return max(best_models, key=lambda x: x[0])[1]

# 3. 汇总所有实验的最新模型路径
def find_latest_models_across_experiments(root_dir="checkpoints_new"):
    """
    遍历所有实验，返回每个实验中最新的模型路径，形式为字典 {实验名: 最新模型路径}
    """
    results = {}
    for exp_name in list_experiments_with_models(root_dir):
        exp_dir = os.path.join(root_dir, exp_name)
        latest_model = find_latest_model_in_experiment(exp_dir)
        if latest_model:
            results[exp_name] = latest_model
    return results


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
def evaluate_model_on_dataset(model_path, dataset_path,cluster_method,pretrained=False):
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

    # model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    extractor = get_feature_extractor(model)

    with torch.no_grad():
        features = extractor(images).squeeze(-1).squeeze(-1).numpy()

    # 聚类
    if cluster_method == 'kmeans':
        n_clusters = len(label_map)
        cluster_model = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = cluster_model.fit_predict(features)

    elif cluster_method == 'dbscan':
        # 你可以根据情况调节 eps/min_samples（或者设成参数传进来）
        cluster_model = DBSCAN(eps=5.0, min_samples=3)
        cluster_labels = cluster_model.fit_predict(features)
        
        # DBSCAN 中 -1 表示噪声点，可选处理：
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        if n_clusters < 1:
            print("[Warning] DBSCAN did not find any clusters.")
            return None

    else:
        raise ValueError(f"Unsupported cluster method: {cluster_method}")

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

    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)

    # 计算每类准确率
    # pred_labels = match_clusters_to_labels(true_labels, cluster_labels)
    # class_acc = {}
    # for i in range(n_clusters):
    #     idx = (true_labels == i)
    #     acc = np.mean(pred_labels[idx] == true_labels[idx])
    #     class_acc[inv_map[i]] = acc

    return ari,nmi

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
    if hasattr(args, "model_path") and args.model_path is not None:
        model_path = args.model_path
        ari,nmi = evaluate_model_on_dataset(model_path, dataset_path,pretrained=args.pretrain)
    elif hasattr(args, "exp_dir") and args.exp_dir:
        model_path = find_latest_model_in_experiment(args.exp_dir)
        ari,nmi = evaluate_model_on_dataset(model_path, dataset_path,pretrained=args.pretrain)
    else:
        models_path = find_latest_models_across_experiments()
        for model_path in models_path:
            ari,nmi = evaluate_model_on_dataset(model_path, dataset_path,pretrained=args.pretrain)
            if ari is not None and nmi is not None:
                print(f"Adjusted Rand Index (ARI): {ari:.4f}")
                print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
            else:
                print("No valid images.")
    print("\n=== Evaluation Results ===")
    if ari is not None and nmi is not None:
        print(f"Adjusted Rand Index (ARI): {ari:.4f}")
        print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    else:
        print("No valid images.")

if __name__ == '__main__':
    main()
