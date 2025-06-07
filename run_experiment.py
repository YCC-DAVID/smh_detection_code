import concurrent.futures
import os

def run_task(param):
    print(f"Running task with {param}")
    os.system(f"python /home/chence/workspace/shm_detection/freezing/smh_detection_code/main_model.py {param}")
    print(f"python /home/chence/workspace/shm_detection/freezing/smh_detection_code/main_model.py {param}")


params = ["-epo 50 -p 4 -ft_epo 5",
          "-p 4 -ft_epo 10 -resume",
          "-p 4 -ft_epo 15 -resume",
          "-p 4 -ft_epo 20 -resume",
          "-ft_epo 5 -resume",
          "-ft_epo 10 -resume",
          "-ft_epo 15 -resume",
          "-ft_epo 20 -resume"]

params2 = ["-epo 60 -p 4 -ft_epo 5",
          "-p 4 -ft_epo 10 -resume",
          "-p 4 -ft_epo 15 -resume",
          "-p 4 -ft_epo 20 -resume",
          "-ft_epo 5 -resume",
          "-ft_epo 10 -resume",
          "-ft_epo 15 -resume",
          "-ft_epo 20 -resume",
          ]

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(run_task, params)

print("First stage tasks are done.")

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(run_task, params2)