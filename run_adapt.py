import concurrent.futures
import os

SRC_PATH = "/home/shared_data/salmonella_detection/AugmentedData/AfricanDataAug"
TAR_PATH = "/home/shared_data/salmonella_detection/OriginalData/AmericanData"
PY_SCRIPT = "/home/chence/workspace/shm_detection/freezing/smh_detection_code/main_adapt.py"
BASE_CKPT = "checkpoints/private1_base_model.pth"

def run_task(cmd):
    print(f" Running: {cmd}")
    os.system(cmd)
    print(f" Done: {cmd}")

base_train_cmd = (
    f"python {PY_SCRIPT} "
    f"--data_root {SRC_PATH} "
    f"--epochs 50 "
    f"--num_classes 2 "
    f"--run_name base_model_african "
)

ft_epochs = [5, 10, 15, 20]
finetune_cmds = [
    f"python {PY_SCRIPT} "
    f"--data_root {TAR_PATH} "
    f"--adapter_only "
    f"--epochs {ep} "
    f"--num_classes 2 "
    f"--load_from {BASE_CKPT} "
    f"--run_name ft_us_{ep}ep"
    for ep in ft_epochs
]


if __name__ == "__main__":

    run_task(base_train_cmd)

    print("=== Base model training complete. Starting finetune tasks... ===")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(run_task, finetune_cmds)

    print("=== 🎉 All finetune tasks complete. ===")
