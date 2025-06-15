import concurrent.futures
import os

def run_task(param):
    print(f"Running task with {param}")
    os.system(f"python /home/chence/workspace/shm_detection/freezing/smh_detection_code/main_model.py {param}")
    print(f"python /home/chence/workspace/shm_detection/freezing/smh_detection_code/main_model.py {param}")


# params = ["-epo 50 --src_path /home/shared_data/salmonella_detection/OriginalData/AfricanData -logger -comb_ds -p 4 --tar_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -ft -ft_p 4",
#           "-epo 50 --src_path /home/shared_data/salmonella_detection/OriginalData/AfricanData -logger -comb_ds -p 3 --tar_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -ft -ft_p 4",
#           "-epo 50 --src_path /home/shared_data/salmonella_detection/OriginalData/AfricanData -logger -comb_ds -p 2 --tar_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -ft -ft_p 4",
#           "-epo 50 --src_path /home/shared_data/salmonella_detection/OriginalData/AfricanData -logger -comb_ds -p 1 --tar_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -ft -ft_p 4",
#           "-epo 50 --src_path /home/shared_data/salmonella_detection/OriginalData/AfricanData -logger -comb_ds -p 0 --tar_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -ft -ft_p 4",
#           "-epo 50 --src_path /home/shared_data/salmonella_detection/OriginalData/AfricanData -logger -comb_ds --tar_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -ft -ft_p 4",
#          ]

params2 = ["-epo 50 --src_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -logger -lr 1e-6",
        #    "-epo 60 --src_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -logger -lr 1e-5",
        #    "-epo 70 --src_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -logger -lr 1e-5",
        #    "-epo 80 --src_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -logger -lr 1e-5",
        #    "-epo 90 --src_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -logger -lr 1e-5",
        #    "-epo 100 --src_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -logger -lr 1e-5",
           "-epo 50 --src_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -logger -lr 2e-6",
           "-epo 50 --src_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -logger -lr 3e-6",
           "-epo 50 --src_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -logger -lr 4e-6",
           "-epo 50 --src_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -logger -lr 5e-6",
           "-epo 50 --src_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -logger -lr 6e-6",
           "-epo 50 --src_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -logger -lr 7e-6",
           "-epo 50 --src_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -logger -lr 8e-6",
           "-epo 50 --src_path /home/shared_data/salmonella_detection/OriginalData/AmericanData -logger -lr 9e-6",
        ]

# with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
#     executor.map(run_task, params)

# print("First stage tasks are done.")

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(run_task, params2)