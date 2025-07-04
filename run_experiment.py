import os
import subprocess
import time
from multiprocessing import Process


# def run_task(param):
#     print(f"Running task with {param}")
#     os.system(f"python /home/chence/workspace/shm_detection/freezing/smh_detection_code/main_model.py {param}")
#     print(f"python /home/chence/workspace/shm_detection/freezing/smh_detection_code/main_model.py {param}")


params  = ["-epo 50 --src_path /home/shared_data/salmonella_detection/AugmentedData/AmericanDataAug --src_name AugAme -logger -p 4 --head linear",
            # "-epo 50 --src_path /home/shared_data/salmonella_detection/AugmentedData/AmericanDataAug --src_name AugAme -logger -p 2",
                

 ]

params1 = ["-epo 50 --src_path /home/shared_data/salmonella_detection/AugmentedData/AmericanDataAug --src_name AugAme -logger -p 2 --head linear",]

# params2 = ["-epo 50 --src_path /home/shared_data/salmonella_detection/OriginalData/AmericanData --src_name OriAme -logger -p 4",
#             "-epo 50 --src_path /home/shared_data/salmonella_detection/OriginalData/AmericanData --src_name OriAme -logger -p 3",
#             "-epo 50 --src_path /home/shared_data/salmonella_detection/OriginalData/AmericanData --src_name OriAme -logger -p 2",
#             "-epo 50 --src_path /home/shared_data/salmonella_detection/OriginalData/AmericanData --src_name OriAme -logger -p 1",

#         ]
params2 = ["-epo 50 --src_path /home/shared_data/salmonella_detection/AugmentedData/AmericanDataAug --src_name AugAme -logger -p 3 --head linear",]

params3 = [#"-epo 50 --src_path /home/shared_data/salmonella_detection/AugmentedData/AmericanDataAug --src_name AugAme -logger -lr 1e-4",
        #    "-epo 50 --src_path /home/shared_data/salmonella_detection/AugmentedData/AmericanDataAug --src_name AugAme -logger -lr 5e-5",
        #    "-epo 50 --src_path /home/shared_data/salmonella_detection/AugmentedData/AmericanDataAug --src_name AugAme -logger -lr 2e-5",
        #    "-epo 50 --src_path /home/shared_data/salmonella_detection/AugmentedData/AmericanDataAug --src_name AugAme -logger -lr 1e-5",
        #    "-epo 50 --src_path /home/shared_data/salmonella_detection/AugmentedData/AmericanDataAug --src_name AugAme -logger -lr 5e-6",
        #    "-epo 50 --src_path /home/shared_data/salmonella_detection/AugmentedData/AmericanDataAug --src_name AugAme -logger -lr 2e-6",
        #    "-epo 50 --src_path /home/shared_data/salmonella_detection/AugmentedData/AmericanDataAug --src_name AugAme -logger -lr 1e-6",
            "-epo 50 --src_path /home/shared_data/salmonella_detection/AugmentedData/AmericanDataAug --src_name AugAme -logger -p 1 --head linear"
]

param_groups = [params, params1, params2, params3]
gpu_assignments = {
    1: [params, params1],  # GPU 0 执行前两组
    3: [params2, params3]  # GPU 1 执行后两组
}

script = "/home/chence/workspace/shm_detection/freezing/smh_detection_code/main_model.py"

def run_param_group(param_group, gpu_id, group_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    for idx, param_str in enumerate(param_group):
        cmd = f"python {script} {param_str}"
        print(f"[GPU {gpu_id}] Group {group_id} Experiment {idx} ➤ {cmd}")
        log_name = f"log_gpu{gpu_id}_group{group_id}_exp{idx}.txt"
        with open(log_name, "w") as log_file:
            subprocess.run(cmd, shell=True, stdout=log_file, stderr=subprocess.STDOUT)
        time.sleep(1)

def run_all():
    processes = []
    for gpu_id, groups in gpu_assignments.items():
        for i, group in enumerate(groups):
            group_id = f"{gpu_id}_{i}"  # 标识是第几组
            p = Process(target=run_param_group, args=(group, gpu_id, group_id))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    run_all()
