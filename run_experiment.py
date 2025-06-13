import concurrent.futures
import os

def run_task(param):
    print(f"Running task with {param}")
    os.system(f"python /home/chence/workspace/shm_detection/freezing/smh_detection_code/main_model.py {param}")
    print(f"python /home/chence/workspace/shm_detection/freezing/smh_detection_code/main_model.py {param}")


params = ["-epo 50 --src_path /home/shared_data/salmonella_detection/AugmentedData/AmericanDataAug -logger -p 4 --tar_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -ft -ft_p 4",
          "-epo 50 --src_path /home/shared_data/salmonella_detection/AugmentedData/AmericanDataAug -logger -p 3 --tar_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -ft -ft_p 4",
          "-epo 50 --src_path /home/shared_data/salmonella_detection/AugmentedData/AmericanDataAug -logger -p 2 --tar_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -ft -ft_p 4",
          "-epo 50 --src_path /home/shared_data/salmonella_detection/AugmentedData/AmericanDataAug -logger -p 1 --tar_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -ft -ft_p 4",
          "-epo 50 --src_path /home/shared_data/salmonella_detection/AugmentedData/AmericanDataAug -logger -p 0 --tar_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -ft -ft_p 4",
          "-epo 50 --src_path /home/shared_data/salmonella_detection/AugmentedData/AmericanDataAug -logger --tar_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -ft -ft_p 4",
         ]

params2 = [
          ]

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(run_task, params)

print("First stage tasks are done.")

# with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
#     executor.map(run_task, params2)