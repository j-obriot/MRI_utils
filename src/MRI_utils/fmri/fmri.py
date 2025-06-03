import numpy as np
import nibabel as nib
import functools
from scipy.signal import savgol_filter

def _nifti_returner(affine, header, arr):
    return nib.Nifti1Image(arr, affine, header)

def _sanitize(obj):
    if type(obj) == nib.Nifti1Image:
        return obj.get_fdata(), functools.partial(_nifti_returner, obj.affine, obj.header)
    elif type(obj) == np.ndarray:
        return obj, lambda a: a
    else:
        try:
            obj = nib.load(obj)
            return _sanitize(obj)
        except:
            pass
        try:
            obj = np.load(obj)
            return _sanitize(obj)
        except:
            pass
        raise TypeError("The object you provided cannot be handled")

def detrend(obj, return_mean=False):
    obj, ret = _sanitize(obj)
    dt = obj - savgol_filter(obj, obj.shape[3], 2, axis=3)
    if return_mean:
        return ret(dt), obj.mean(3)
    return ret(dt)


def tsnr(obj, detrend=False):
    obj, ret = _sanitize(obj)
    mean = obj.mean(3)
    if detrend:
        obj = detrend(obj)
    return ret(mean / obj.std(3))

def save_first_volume(filename, out_filename=None):
    if out_filename is None:
        out_filename = filename
    img = nib.load(filename)
    nib.save(nib.Nifti1Image(img.get_fdata(), img.affine, img.header), out_filename)
