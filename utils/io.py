import skimage.io
import numpy as np
from PIL import Image

def imread(path, make_8bit = False, rescale = False, rescale_min = False, rescale_max = False):
    im_arr = Image.open(path)#skimage.io.imread(path)
    #im_arr = im_arr.resize((720,720))
    im_arr = np.array(im_arr)
    if im_arr.dtype == np.uint8:
        dtype = 'uint8'
    elif im_arr.dtype == np.uint16:
        dtype = 'uint16'
    elif im_arr.dtype in [np.float16, np.float32, np.float64]:
        if np.amax(im_arr) <= 1 and np.amin(im_arr) >= 0:
            dtype = 'zero-one normalized'  # range = 0-1
        elif np.amax(im_arr) > 0 and np.amin(im_arr) < 0:
            dtype = 'z-scored'
        elif np.amax(im_arr) <= 255:
            dtype = '255 float'
        elif np.amax(im_arr) <= 65535:
            dtype = '65535 float'
        else:
            raise TypeError('The loaded image array is an unexpected dtype.')
    else:
        raise TypeError('The loaded image array is an unexpected dtype.')
    
    return im_arr

def _check_channel_order(im_arr, framework = 'torch'):
    im_shape = im_arr.shape
    if len(im_shape) == 3:
        if im_shape[0] > im_shape[2] and framework in ['torch', 'pytorch']:
            im_arr = np.moveaxis(im_arr, 2, 0)
    elif len(im_shape) == 4:
        if im_shape[1] > im_shape[3] and framework in ['torch', 'pytorch']:
            im_arr = np.moveaxis(im_arr, 3, 1)

    return im_arr