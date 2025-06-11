import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
from torchvision import models, datasets, transforms
from trainer import build_dataloaders, Trainer
from models import build_model
import wandb
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str, required=True)  # 源数据集路径
parser.add_argument('--tar_path', type=str, default=None)  # 目标数据集路径，只有在 adapter_only 时需要
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--finetune_only', action='store_true')
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--run_name', type=str, default=None)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--load_from', type=str, default=None, help='加载训练好的基础模型')

parser.add_argument('--tuning_method', type=str, default='prompt')
parser.add_argument('--prompt_size', default=10, type=int, help='prompt size')
parser.add_argument('--kernel_size', default=3, type=int)
parser.add_argument('--adapt_size', default=8, type=float)
parser.add_argument('--adapt_scale', default=1.0, type=float)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def freeze_backbone_except_adapter(model):
    for name, param in model.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False

def load_checkpoint(model, optimizer, scheduler, path):
    if os.path.exists(path):
        ckpt = torch.load(path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        print(f"恢复模型: {path}")
    else:
        print(f"未找到恢复文件: {path}")

def load_base_weights(model, path):
    if os.path.exists(path):
        print(f"加载基础模型参数自: {path}")
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        print(f"加载失败，路径无效: {path}")

def main():
    args = parser.parse_args()
    src_path = args.src_path
    tar_path = args.tar_path
    num_classes = args.num_classes
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    ft_only = args.finetune_only
    run_name = args.run_name
    resume = args.resume
    load_from = args.load_from


    wandb.init(
        project="conv-adapter-finetune",
        name=str(run_name) if run_name else ("adapter_ft" if ft_only else "full_ft"),
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "finetune_only": ft_only,
            "model": "resnet50_conv_adapter"
        },
        settings=wandb.Settings(init_timeout=300)
    )

    # 使用修改后的数据加载器
    generator = torch.Generator().manual_seed(42)

    # 判断是否是 adapter_only 模式
    if ft_only:
        if not tar_path or not os.path.exists(tar_path):
            raise ValueError("需要提供有效的 --tar_path 以在 adapter_only 模式下进行微调")
        # 目标数据集微调
        train_loader, val_loader, ft_loader, ft_val_loader = build_dataloaders(src_path, tar_path, batch_size, generator)
        train_loader = ft_loader
        val_loader = ft_val_loader
    else:
        if not os.path.exists(src_path):
            raise ValueError("需要提供有效的 --src_path 以进行基础模型训练")
        # 源数据集训练基础模型
        train_loader, val_loader, _, _ = build_dataloaders(src_path, None, batch_size, generator)

    model =  models.resnet50(pretrained=True) #build_model(model_name='resnet50', pretrained=True,num_classes=2,tuning_method=args.tuning_method,args=args)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)


    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
        )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir="adapt_checkpoints",
        logger=wandb,
        status="finetuning" if ft_only else "full training"
    )

    if resume:
        resume_path = "adapt_checkpoints/last_checkpoint.txt"
        if os.path.exists(resume_path):
            with open(resume_path, "r") as f:
                ckpt_path = f.read().strip()
                load_checkpoint(model, optimizer, scheduler, ckpt_path)

    trainer.train(epochs)

    #  基础模型保存
    if not ft_only:
        with open("adapt_checkpoints/last_checkpoint.txt", "r") as f:
            best_path = f.read().strip()
        shutil.copy(best_path, "checkpoints/private1_base_model.pth")
        print(" 基础模型已保存为 checkpoints/private1_base_model.pth")

    wandb.finish()

if __name__ == "__main__":
    main()
