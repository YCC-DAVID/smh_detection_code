import concurrent.futures
import os

def run_task(param):
    print(f"Running task with {param}")
    os.system(f"python /home/chence/workspace/shm_detection/freezing/smh_detection_code main_model.py {param}")
    print(f"python /home/chence/workspace/shm_detection/freezing/smh_detection_code main_model.py {param}")


params = ["-p 3 -ft_epo 5 -resume",
          "-p 3 -ft_epo 10 -resume",
          "-p 3 -ft_epo 15 -resume",
          "-p 3 -ft_epo 20 -resume",
          "-p 4 -ft_epo 5 -resume",
          "-p 4 -ft_epo 10 -resume",
          "-p 4 -ft_epo 15 -resume",
          "-p 4 -ft_epo 20 -resume",
          "-p 2 -ft_epo 5 -resume",
          "-p 2 -ft_epo 10 -resume",
          "-p 2 -ft_epo 15 -resume",
          "-p 2 -ft_epo 20 -resume"
          ]
# params = ["-p 9 -fzepo 210"]
# ,"-p 7 -fzepo 100"]
        #   "-p 7 -fzepo 130",
        #   "-p 7 -fzepo 150",
        #   "-p 8 -fzepo 130",
        #   "-p 8 -fzepo 150",
        #   "-p 8 -fzepo 170",
        #   "-p 9 -fzepo 100",
        #   "-p 9 -fzepo 150"]
        #   "-p 3 4 -fzepo 0 193",
        #   "-p 3 5 6 -fzepo 10 185 195",
        #   "-p 0 3 4 6 -fzepo 0 20 170 183",
        #   "-p 0 3 6 -fzepo 0 30 169",
        #   "-p 3 6 -fzepo 1 195"]
# 使用线程池执行任务，每次最多运行 3 个线程
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(run_task, params)

print("All tasks are done.")