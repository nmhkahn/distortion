import skimage.io
import skimage.util
import skimage.filters
import skimage.transform
import numpy as np
from PIL import Image

def gaussian_noise(im, var=0.01):
    # var: 0 ~ 0.1
    noisy = skimage.util.random_noise(im, mode="gaussian", var=var)
    noisy = np.clip(noisy, 0, 1.0)
    return noisy


def gaussian_blur(im, sigma=5):
    # sigma: 0 ~ 10
    noisy = skimage.filters.gaussian(im, sigma=sigma)
    noisy = np.clip(noisy, 0, 1.0)
    return noisy


def quantization_noise(im, level=16):
    # level: 32 ~ 2
    im = skimage.util.img_as_ubyte(im)
    level = int(level)
    
    T = np.arange(0, 255, 256/level)
    noisy = np.digitize(im.flat, T)
    noisy = T[noisy-1].reshape(im.shape).astype(np.uint8)

    noisy = skimage.util.img_as_float(noisy)
    noisy = np.clip(noisy, 0, 1.0)
    return noisy


def low_resolution(im, scale=0.2):
    # scale: 0.1 ~ 0.5
    size   = im.shape
    scaled = skimage.transform.rescale(im, scale)
    noisy  = skimage.transform.resize(scaled, size)
    noisy = np.clip(noisy, 0, 1.0)
    return noisy


def f_noise(im, scale=5, clip=True):
    # scale: 1 ~ 15
    def one_f(beta=-1):
        dim = im.shape

        u1 = np.arange(np.floor(dim[0]/2)+1)
        u2 = -1 * np.arange(np.ceil(dim[0]/2)-1, 0, -1)
        u = np.concatenate([u1, u2]) / dim[0]
        u = np.tile(u, (dim[1], 1))
        u = np.swapaxes(u, 0, 1)

        v1 = np.arange(np.floor(dim[1]/2)+1)
        v2 = -1 * np.arange(np.ceil(dim[1]/2)-1, 0, -1)
        v = np.concatenate([v1, v2]) / dim[1]
        v = np.tile(v, (dim[0], 1))

        s_f = np.power(np.power(u, 2) + np.power(v, 2) + 1e-5, beta/2)
        s_f[s_f == np.inf] = 0

        phi = np.random.uniform(size=dim)

        x = np.power(s_f, 0.5) * (np.cos(2*np.pi*phi) + 1j*np.sin(2*np.pi*phi))
        x = np.fft.ifft2(x)
        x = np.real(x)
        return x

    im = skimage.util.img_as_float(im)
    noisy = im.copy()
    noisy[:,:] = im[:,:] + scale*one_f(-2)

    noisy = np.clip(noisy, 0, 1.0)
    return noisy


def jpeg_compression(im, quality=30, filename="tmp"):
    quality = int(quality)

    im = skimage.util.img_as_ubyte(im)
    obj = Image.fromarray(im)
    obj.save("./{}.jpg".format(filename), format="JPEG", quality=int(quality))
    noisy = skimage.io.imread("./{}.jpg".format(filename))
    return noisy
