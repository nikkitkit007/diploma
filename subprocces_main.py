import os
import subprocess
import time

input_dir = '/home/nikita/Desktop/one-pixel-attack-master/cifar10/test'
output_dir = '/home/nikita/Desktop/one-pixel-attack-master/output'

MAXPROC = 8

if __name__ == '__main__':
    ls = os.listdir('cifar10/test/')

    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    procs = []
    for i in range(MAXPROC):
        for filename in ls[i::MAXPROC]:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            p = subprocess.Popen(['python', 'non_targeted.py', '--config', "config.yaml", "--input", input_path])
            procs.append(p)
            while len(procs) == MAXPROC:
                print([pr.pid for pr in procs])
                for p in procs:
                    if p.poll() is not None:
                        procs.remove(p)
                time.sleep(3)
