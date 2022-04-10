import os
sf = 2

root = f'../data/Set5/image_SRF_{sf}'

fns = os.listdir(root)
img_names = set()
for fn in fns:
    if not fn.endswith('.png'): continue
    img_names.add('_'.join(fn.split('_')[:-1]))

def imread(path):
    from PIL import Image
    import numpy as np
    img = Image.open(path)
    img = np.array(img) / 255
    return img
    
from skimage.metrics import peak_signal_noise_ratio
print(img_names)
import numpy as np
import cv2
from torchlight.transforms import Resize

def cal_psnr(hr, sr, scale):
    hr = hr[scale:-scale, scale:-scale, :]
    sr = sr[scale:-scale, scale:-scale, :]
    
    gray_coeffs = np.array([65.738, 129.057, 25.064])[None][None] / 256
    hr = np.sum(hr*gray_coeffs, axis=2)
    sr = np.sum(sr*gray_coeffs, axis=2)
    return peak_signal_noise_ratio(hr, sr, data_range=1)
    
total_psnr = 0
for name in img_names:
    hr = os.path.join(root, name + '_HR.png')
    sr = os.path.join(root, name + '_bicubic.png')
    lr = os.path.join(root, name + '_LR.png')
    
    hr = imread(hr)
    lr = Resize(1/sf)(hr)
    
    # lr = imread(lr)
    sr = imread(sr)
    sr = Resize(sf)(lr)
    
    psnr = cal_psnr(hr, sr, sf)
    total_psnr += psnr

print(total_psnr / len(img_names))