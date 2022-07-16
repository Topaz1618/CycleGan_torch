import pickle
import gzip
import cv2
import torch
import numpy
from PIL import Image, ImageFilter
from numpy import (amin, amax, ravel, asarray, arange, ones, newaxis,
                   transpose, iscomplexobj, uint8, issubdtype, array)
import numpy as np

#from scipy.misc import imresize
from matplotlib import pyplot
from torchvision import datasets
from torchvision import transforms

# print(PATH/FILENAME)
resize_scale = 286

def data_load(path, subfolder, transform, batch_size, shuffle=False):
    dset = datasets.ImageFolder(path, transform)
    ind = dset.class_to_idx[subfolder]
    n = 0
    for i in range(dset.__len__()):
        if ind != dset.imgs[n][1]:
            del dset.imgs[n]
            n -= 1

        n += 1

    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle)


def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    if data.dtype == uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(uint8)


def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):

    data = asarray(arr)
    if iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = numpy.flatnonzero(asarray(shape) == 3)[0]
        else:
            ca = numpy.flatnonzero(asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image


def imgs_resize(imgs, resize_scale = 286):

    """	
    ============= IDX: 0 Resize Img type: <class 'torch.Tensor'> ===============  
	Before: (3, 256, 256) Type:<class 'numpy.ndarray'>
	After:(286, 286, 3) Type:<class 'numpy.ndarray'>
	torch.Size([1, 3, 286, 286])
	torch.Size([1, 3, 256, 256])

    """
    outputs = torch.FloatTensor(imgs.size()[0], imgs.size()[1], resize_scale, resize_scale)
    for i in range(imgs.size()[0]):
    	# source code
        #img = imresize(imgs[i].numpy(), [resize_scale, resize_scale])
        
        # not work
        #img = np.moveaxis(imgs[i].numpy(), 0, 2)
        #img = cv2.resize(img, [resize_scale, resize_scale])
        
        # not work
        #img = np.transpose(imgs[i].numpy())
        #img = cv2.resize(img, [286, 286])
        
        # m2
        pil_image = toimage(imgs[i].numpy(), mode=None)
        img = cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)
        print(type(img))
        img = cv2.resize(img, [286, 286])
        
        cv2.imwrite("a.jpg", img)
        
	
        print(f"============= IDX: {i} Resize Img type: {type(imgs[i])} ===============  ")
        print(f"Before: {imgs[i].numpy().shape} Type:{type(imgs[i].numpy())}")
        print(f"After:{img.shape} Type:{type(img)}")
        #print(f"{img.transpose(2, 0, 1).shape}")
        
        
        outputs[i] = torch.FloatTensor((img.transpose(2, 0, 1).astype(np.float32).reshape(-1, imgs.size()[1], resize_scale, resize_scale) - 127.5) / 127.5)
	
    return outputs

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


train_loader_A = data_load("/home/topaz/MacHome/Projects/py_cyclegan/data/horse2zebra", "trainA", transform, 1, shuffle=True)
train_loader_B = data_load("/home/topaz/MacHome/Projects/py_cyclegan/data/horse2zebra", "trainB", transform, 1, shuffle=True)

for (realA, _), (realB, _) in zip(train_loader_A, train_loader_B):
    realA = imgs_resize(realA, resize_scale)
    print(realA.shape)
    print(realB.shape)
    
    
    
