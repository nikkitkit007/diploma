import os
import subprocess
import time

from configurations import CONFIG, origin_dataset_path, broken_dataset_path

input_dir = origin_dataset_path
output_dir = broken_dataset_path

max_proc = CONFIG["max_proc"]


if __name__ == '__main__':
    ls = os.listdir(input_dir)

    procs = []
    for i in range(max_proc):
        for filename in ls[i::max_proc]:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            p = subprocess.Popen(['python3', 'non_targeted.py', '--config', "config.yaml", "--input", input_path])
            procs.append(p)
            while len(procs) == max_proc:
                print([pr.pid for pr in procs])
                for p in procs:
                    if p.poll() is not None:
                        procs.remove(p)
                time.sleep(3)
