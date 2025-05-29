import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.multiprocessing import freeze_support

def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    for images, _ in tqdm(loader, desc="Computing mean/std"):
        batch_samples = images.size(0)  # batch size (64)
        images = images.view(batch_samples, 3, -1)  # (B, 3, H*W)
        mean += images.mean(2).sum(0)  # (3,)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images
    return mean, std

# 用未标准化的 transform


if __name__ == '__main__':
    from torch.multiprocessing import freeze_support
    freeze_support()  # 可选，兼容 frozen 程序

    raw_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 尺寸和你训练时一致
    transforms.ToTensor()
    ])


    # 加载数据集
    dataset = ImageFolder('AmericanData', transform=raw_transform)
    
    mean, std = compute_mean_std(dataset)
    print("Mean:", mean)
    print("Std:", std)