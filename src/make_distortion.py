import os
import glob
import argparse

import numpy as np
import scipy.misc
from multiprocessing import Process, Queue

from distortion import *

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_test",
                        type=str)
    parser.add_argument("--num_process",
                        type=int,
                        default=4)
    return parser.parse_args()


def do_work(im_paths, args, fn, quality, qq):
    pid = os.getpid()

    fn_name = fn.__name__
    if fn_name == "gaussian_noise":
        dirname = "gwn"
    elif fn_name == "salt_pepper_noise":
        dirname = "snp"
    elif fn_name == "quantization_noise":
        dirname = "quant"
    elif fn_name == "jpeg_compression":
        dirname = "jpeg"
    elif fn_name == "f_noise":
        dirname = "fnoise"
    elif fn_name == "gaussian_blur":
        dirname = "gblur"

    for step, path in enumerate(im_paths):
        im = scipy.misc.imread(path)
        
        im_name = path.split("/")[-1].split(".")[0]
        new_path = "{}/{}/Q{}/{}.png".format(args.train_test, dirname, qq, im_name)
        
        try:
            if fn_name == "jpeg_compression":
                dim = fn(im, quality, pid)
            else:
                dim = fn(im, quality)
        except error:
            print(path)
        scipy.misc.imsave(new_path, dim)

        if (step+1) % 1000 == 0:
            print("process {} finished {}/{} steps"
                  .format(pid, step+1, len(im_paths)))


def distribute(args, fn, quality, qq):
    print("{}-{}".format(fn.__name__, qq))

    im_paths = glob.glob("{}/color/*.jpg".format(args.train_test))
    num_proc = args.num_process
    works_per_proc = int(len(im_paths)/num_proc)
    
    works = [[works_per_proc*i, works_per_proc*(i+1)] for i in range(num_proc)]
    works[-1][1] = len(im_paths)
    
    pool = []
    for i in range(num_proc):
        start, end = works[i]
        job = im_paths[start:end]
        proc = Process(target=do_work, args=(job, args, fn, quality, qq))
        proc.start()
        pool.append(proc)
    
    for p in pool:
        p.join()


def main(args):
    todos = [
        [f_noise, 8, 10, 12, 14, 16],
        [gaussian_noise, 0.01, 0.015, 0.02, 0.025, 0.03],
        [salt_pepper_noise, 0.01, 0.02, 0.03, 0.04, 0.05],
        [gaussian_blur, 1.5, 1.75, 2.0, 2.25, 2.5],
        [quantization_noise, 12, 10, 8, 6, 4],
        [jpeg_compression, 50, 40, 30, 20, 10],
    ]
    
    qqs = [50, 40, 30, 20, 10]
    for todo in todos:
        fn = todo[0]
        for i, q in enumerate(todo[1:]):
            distribute(args, fn, q, qqs[i])


if __name__ == "__main__":
    args = parse_args()
    main(args)
