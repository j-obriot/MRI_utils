import numpy as np
import nibabel as nib
import pandas as pd
import functools
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from nilearn.maskers import NiftiSpheresMasker
from nilearn.glm.first_level import (
    FirstLevelModel,
    make_first_level_design_matrix,
)

def _nifti_returner(affine, header, arr):
    return nib.Nifti1Image(arr, affine, header)

def _sanitize(obj, firstpass=True):
    if type(obj) == nib.Nifti1Image:
        return obj.get_fdata(), functools.partial(_nifti_returner, obj.affine, obj.header)
    elif type(obj) == np.ndarray:
        return obj, lambda a: a
    elif firstpass:
        try:
            obj = nib.load(obj)
            return _sanitize(obj, False)
        except:
            pass
        try:
            obj = np.load(obj)
            return _sanitize(obj, False)
        except:
            pass
    raise TypeError("The object you provided cannot be handled")

def detrend(obj, return_mean=False):
    obj, ret = _sanitize(obj)
    dt = obj - savgol_filter(obj, obj.shape[3], 2, axis=3)
    if return_mean:
        return ret(dt), obj.mean(3)
    return ret(dt)

def smooth(obj, sigma=0.75):
    obj, ret = _sanitize(obj)
    return ret(gaussian_filter(obj, sigma, axes=(0, 1, 2)))

def get_nvolumes(obj, n, start=None):
    obj, ret = _sanitize(obj)
    if start is None:
        return ret(obj[..., -n:])
    else:
        return ret(obj[..., start:n+start])

def tsnr(obj, detrend=False):
    obj, ret = _sanitize(obj)
    mean = obj.mean(-1)
    if detrend:
        obj = detrend(obj)
    return ret(mean / obj.std(-1))

def save_first_volume(filename, out_filename=None):
    if out_filename is None:
        out_filename = filename
    img = nib.load(filename)
    nib.save(nib.Nifti1Image(img.get_fdata(), img.affine, img.header), out_filename)

def get_reasonable_confounds(file, nvol, index=None):
    confounds = pd.read_csv(file, sep='\t')
    alen = 0
    max_acomp = 15
    for i in range(max_acomp):
        if f"a_comp_cor_{i:02}" not in confounds:
            break
        alen += 1
    keys = [*[f"trans_{i}" for i in ['x', 'y', 'z']],
            *[f"trans_{i}_power2" for i in ['x', 'y', 'z']],
            *[f"trans_{i}_derivative1" for i in ['x', 'y', 'z']],
            *[f"trans_{i}_derivative1_power2" for i in ['x', 'y', 'z']],
            *[f"rot_{i}" for i in ['x', 'y', 'z']],
            *[f"rot_{i}_power2" for i in ['x', 'y', 'z']],
            *[f"rot_{i}_derivative1" for i in ['x', 'y', 'z']],
            *[f"rot_{i}_derivative1_power2" for i in ['x', 'y', 'z']],
            *[f"a_comp_cor_{i:02}" for i in range(alen)],
            "framewise_displacement",
            ]
    return get_confounds(file, keys, nvol, index)

def get_confounds(file, keys, nvol, index=None):
    confounds = pd.read_csv(file, sep='\t')
    mat = pd.DataFrame()
    for k in reversed(keys):
        mat.insert(0, k, confounds[-nvol:][k])
    if index is not None:
        mat.set_index(index)
    return mat

class _has_changed:
    def __init__(self, obj, **kwargs):
        self.obj = obj
        self.hash = hash(frozenset(kwargs.items()))

    def did_it(self, **kwargs):
        return self.hash == hash(frozenset(kwargs.items()))

class Func:
    def __init__(self, volumes, confounds, confounds_keys=None, nvols=0, skip_vols=0, t_r=2):
        if nvols == 0:
            nvols = _sanitize(volumes)[0].shape[-1] - skip_vols
        self.nvols = nvols
        self.volumes = get_nvolumes(volumes, nvols, start=skip_vols)
        self.detrended, self.mean = detrend(self.volumes, return_mean=True)
        self.index = np.linspace(0, t_r*(nvols-1), nvols)
        if confounds_keys is None:
            self.confounds = get_reasonable_confounds(confounds, nvols, index=self.index)
        else:
            self.confounds = get_confounds(confounds, confounds_keys, nvols, index=self.index)

        self.t_r = t_r
        self._tsnr = None
        self.smoothed = None
        self.cleaned = None
        self.seed = None
        self.dmn = None

    def smooth(self, sigma):
        self.smoothed = smooth(self.detrended, sigma)

    def tsnr(self):
        if self._tsnr is None:
            obj, ret = _sanitize(self.detrended)
            self._tsnr = ret(self.mean / obj.std(-1))
        return self._tsnr

    def set_confounds(self, confounds, confounds_keys=None):
        if confounds_keys is None:
            self.confounds = get_reasonable_confounds(confounds, self.nvols, index=self.index)
        else:
            self.confounds = get_confounds(confounds, confounds_keys, self.nvols, index=self.index)

    def seed_based_DMN(self, seed=(0, -53, 26)):
        use = self.detrended
        if self.smoothed is not None:
            use = self.smoothed

        args = {
                "vols": use,
                "seed": seed,
                "confounds": self.confounds,
                }
        if self.dmn is not None and self.dmn.did_it(**args):
            return self.dmn.obj

        if self.seed is None or self.seed.did_it(seed = seed):
            seed_masker = NiftiSpheresMasker(
                [seed],
                radius=4,
                detrend=False,
                standardize="zscore_sample",
                t_r=self.t_r,
                low_pass=0.1,
                high_pass=0.01,
                memory="nilearn_cache",
                memory_level=1,
                verbose=0,
            )
            self.seed = _has_changed(seed_masker.fit_transform(use), seed = seed)

        design_matrix = make_first_level_design_matrix(
                self.index,
                hrf_model="spm",
                add_regs=self.seed.obj,
                add_reg_names=["pcc_seed"],
                )
        for k in self.confounds.keys():
            design_matrix.insert(1, k, self.confounds[k])

        dmn_contrast = np.array([1] + [0] * (design_matrix.shape[1] - 1))

        contrasts = {"seed_based_glm": dmn_contrast}
        first_level_model = FirstLevelModel(t_r=self.t_r, slice_time_ref=0.)
        first_level_model = first_level_model.fit(
            run_imgs=use, design_matrices=design_matrix
        )

        self.dmn = _has_changed(first_level_model.compute_contrast(
                contrasts["seed_based_glm"], output_type="z_score"
        ), **args)

        return self.dmn.obj
