import os
import glob

import numpy as np
import scipy.misc
from PIL import Image
from distortion import *

def do_work(im_paths, fn, quality, quality_factor):
    pid = os.getpid()

    fn_name = fn.__name__
    if fn_name == "gaussian_noise":
        dirname = "gwn"
    elif fn_name == "low_resolution":
        dirname = "lr"
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

        new_path = "result/{}_{}_{}.jpg".format(dirname, quality_factor, im_name)
        
        if fn_name == "jpeg_compression":
            dim = fn(im, quality, 1)
        else:
            dim = fn(im, quality) * 255
            dim = dim.astype(np.uint8)
        
        im = Image.fromarray(dim)
        im.save(new_path, format="JPEG", quality=100)

def main():
    todos = [
        [f_noise, 4, 5, 6, 7, 8],
        [gaussian_noise, 0.005, 0.0075, 0.01, 0.0125, 0.015],
        [gaussian_blur, 1, 1.5, 2.0, 2.25, 2.5],
        [low_resolution, 0.5, 0.4, 0.35, 0.3, 0.25],
        [quantization_noise, 12, 11, 10, 9, 8],
        [jpeg_compression, 50, 40, 30, 20, 10],
    ]
    ims = glob.glob("original/*.jpg")
    ims = [ims[0]]
    for todo in todos:
        for i, q in enumerate(todo[1:]):
            do_work(ims, todo[0], q, i+1)


if __name__ == "__main__":
    main()
