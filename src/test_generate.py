import os
import glob
import scipy.misc
import numpy as np
from scipy.stats import truncnorm
from PIL import Image
from distortion import *

FN_NAME = {
    "gaussian_noise":     "gwn", 
    "snp": "snp",
    "quantization_noise": "quant",
    "jpeg_compression":   "jpeg",
    "gaussian_blur":      "gblur",
    "denoising":          "denoising",
    "low_resolution":     "low_res",
    "f_noise":            "fnoise",
}

TYPE = [
    [gaussian_noise, 0.005, 0.02],
    [snp, 0.005, 0.02],
    [quantization_noise, 16, 8],
    [denoising, 0.01, 0.1],
    [jpeg_compression, 50, 10],
    [gaussian_blur, 1, 2.5],
    [low_resolution, 0.5, 0.2],
    [f_noise, 6, 10],
    [None, 0, 0]
]


def constrained_sum_sample(n, total):
    mean_ = total / n
    low_ = mean_*0.7
    upp_ = mean_*1.3
    
    def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
        return truncnorm(
                (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    dividers = get_truncated_normal(mean=mean_, sd=n*3, low=low_, upp=upp_).rvs(n-1)
    dividers = np.array(sorted([int(d) for d in dividers]))
    dividers = [np.sum(dividers[:i+1]) for i in range(len(dividers))]

    return [0] + dividers + [total]


def _horizontal(im, num_parts):
    h, w = im.shape[:2]
    if len(im.shape) == 3:
        new_im = np.empty((h, w, 3), dtype=np.uint8)
    else:
        new_im = np.empty((h, w), dtype=np.uint8)
    parts = constrained_sum_sample(num_parts, h)
    
    distorts = list()
    for i in range(1, len(parts)):
        idx = np.random.randint(len(TYPE))
        type_ = TYPE[idx][0]
        min_, max_  = TYPE[idx][1:]
        if type_ is None: continue
        level_ = np.around(np.random.uniform(min_, max_), 3)

        start, end = parts[i-1], parts[i]
        new_im[start:end, :] = type_(im, level_)[start:end, :] 
        
        print(level_)
        info = {
            "ymin": start, "xmin": 0,
            "ymax": end,   "xmax": len(im),
            "type": FN_NAME[type_.__name__], "level": "{:.3f}".format(level_)
        }
        distorts.append(info)
    
    return new_im, distorts


def _vertical(im, num_parts):
    h, w = im.shape[:2]
    if len(im.shape) == 3:
        new_im = np.empty((h, w, 3), dtype=np.uint8)
    else:
        new_im = np.empty((h, w), dtype=np.uint8)
    parts = constrained_sum_sample(num_parts, w)

    distorts = list()
    for i in range(1, len(parts)):
        idx = np.random.randint(len(TYPE))
        type_ = TYPE[idx][0]
        min_, max_  = TYPE[idx][1:]
        if type_ is None: continue
        level_ = np.around(np.random.uniform(min_, max_), 3)
        
        start, end = parts[i-1], parts[i]
        new_im[:, start:end] = type_(im, level_)[:, start:end]
    
        print(level_)
        info = {
            "ymin": 0,       "xmin": start,
            "ymax": len(im), "xmax": end,
            "type": FN_NAME[type_.__name__], "level": "{:.3f}".format(level_)
        }
        distorts.append(info)
    
    return new_im, distorts


def _block(im, num_parts):
    h, w = im.shape[:2]
    if len(im.shape) == 3:
        new_im = np.empty((h, w, 3), dtype=np.uint8)
    else:
        new_im = np.empty((h, w), dtype=np.uint8)
    h_parts = constrained_sum_sample(num_parts, h)
    w_parts = constrained_sum_sample(num_parts, w)

    distorts = list()
    for i in range(1, len(h_parts)):
        for j in range(1, len(w_parts)):
            idx = np.random.randint(len(TYPE))
            type_ = TYPE[idx][0]
            min_, max_  = TYPE[idx][1:]
            if type_ is None: continue
            level_ = np.around(np.random.uniform(min_, max_), 3)

            start_h, end_h = h_parts[i-1], h_parts[i]
            start_w, end_w = w_parts[j-1], w_parts[j]
            new_im[start_h:end_h, start_w:end_w] = type_(im, level_)[start_h:end_h, start_w:end_w]
            
            info = {
                "ymin": start_h, "xmin": start_w,
                "ymax": end_h,   "xmax": end_w,
                "type": FN_NAME[type_.__name__], "level": "{:.3f}".format(level_)
            }
            distorts.append(info)
    
    return new_im, distorts
           

def generate(paths):
    for step, path in enumerate(paths):
        im = scipy.misc.imread(path)
        fname = path.split("/")[-1].split(".")[0]
        
        i = 0
        for scenario in [_horizontal, _vertical, _block]:
            for num_parts in range(2, 6):
                new_im, distorts = scenario(im, num_parts)
                new_im = Image.fromarray(new_im)
                
                in_path = path.replace("reference", "distorted")
                in_path = in_path.split(".")[0] + "_{}.jpg".format(i+1)

                info = {
                    "path": in_path,
                    "ref_path": path,
                    "objects": distorts
                }
                print(info)

                i += 1   
                new_im.save(in_path, format="JPEG", quality=100)


def main():
    paths = glob.glob("flickr/reference/train/gray/animal_1.jpg")
    generate(paths)


if __name__ == "__main__":
    main()
