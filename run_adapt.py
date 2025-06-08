import concurrent.futures
import os

# 路径设置
SRC_PATH = "/home/shared_data/salmonella_detection/AugmentedData/AfricanDataAug"
TAR_PATH = "/home/shared_data/salmonella_detection/OriginalData/AmericanData"
PY_SCRIPT = "/home/chence/workspace/shm_detection/freezing/smh_detection_code/main_adapt.py"
BASE_CKPT = "checkpoints/private1_base_model.pth"

# 运行任务的函数
def run_task(cmd):
    print(f" Running: {cmd}")
    os.system(cmd)
    print(f" Done: {cmd}")

# 基础模型训练命令
base_train_cmd = (
    f"python {PY_SCRIPT} "
    f"--src_path {SRC_PATH} "
    f"--epochs 50 "
    f"--num_classes 2 "
    f"--run_name base_model_african "
)

# 微调 epochs 设置（不同的训练轮数）
ft_epochs = [5, 10, 15, 20]
finetune_cmds_adapter_only = [
    f"python {PY_SCRIPT} "
    f"--src_path {SRC_PATH} "
    f"--tar_path {TAR_PATH} "
    f"--adapter_only "
    f"--epochs {ep} "
    f"--num_classes 2 "
    f"--load_from {BASE_CKPT} "
    f"--run_name ft_us_adapter_only_{ep}ep"
    for ep in ft_epochs
]

finetune_cmds_full_model = [
    f"python {PY_SCRIPT} "
    f"--src_path {SRC_PATH} "
    f"--tar_path {TAR_PATH} "
    f"--epochs {ep} "
    f"--num_classes 2 "
    f"--load_from {BASE_CKPT} "
    f"--run_name ft_us_full_model_{ep}ep"
    for ep in ft_epochs
]

# 主程序
if __name__ == "__main__":

    # 执行基础模型训练
    print("=== Starting base model training... ===")
    run_task(base_train_cmd)

    print("=== Base model training complete. Starting finetune tasks... ===")

    #  并行执行 adapter-only 微调任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(run_task, finetune_cmds_adapter_only)

    print("=== All adapter-only finetune tasks complete. ===")

    #  并行执行完整模型微调任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(run_task, finetune_cmds_full_model)

    print("===  All full model finetune tasks complete. ===")
