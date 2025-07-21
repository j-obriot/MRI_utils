import numpy as np
import torch
from math import pi
from tqdm import tqdm
from ggrappa import GRAPPA_Recon
import scipy.interpolate as interpolate
from tqdm.contrib.concurrent import thread_map

def forward(data, dim):
    return np.fft.ifftshift(
                np.fft.ifftn(
                    np.fft.fftshift(data, axes=dim),
                    axes=dim),
                axes=dim)

def backward(img, dim):
    return np.fft.ifftshift(
                np.fft.fftn(
                    np.fft.fftshift(img, axes=dim),
                    axes=dim),
                axes=dim)

def zeropad(data, ref=None, shape=None):
    if type(ref) == type(None) and type(shape) == type(None):
        return data
    # what to do if no ref? open question.
    if type(ref) != None:
        shape = ref.shape
    diff = np.array([a - b for a, b in zip(shape, data.shape)])
    left = diff // 2
    right = diff - left

    return np.pad(data, [(l, r) for l, r in zip(left, right)], constant_values=0)

def extract_data_acs(img, dim=[], af=1, nb_lines=24, freq=True, acs_shift=[]):
    """
    Extracts Auto-Calibration Signal (ACS) lines from a full image and modifies the image data based on the acceleration factor.

    Parameters:
        img (ndarray): The input image data.
        dim (int or tuple of ints): The dimensions along which to extract data and perform FFT.
        af (int or list of ints, optional): Acceleration factor(s) for undersampling. Default is 1.
        nb_lines (int, optional): Number of ACS lines to extract. Default is 24.
        freq (bool, optional): If True, assumes input is in frequency domain; otherwise, performs FFT. Default is True.
        acs_shift (list of ints, optional): Shifts the position of ACS lines along each dimension. Default is an empty list, which implies no shift.

    Returns:
        tuple: A tuple containing:
            - data_full (ndarray): The modified image data with undersampling applied.
            - acs (ndarray): The extracted ACS lines.
    """
    if type(af) == int:
        af = [af]
    if len(dim) < 1:
        dim = range(len(img.shape))
    if len(acs_shift) < 1:
        acs_shift = [0] * len(dim)
    
    if not freq:
        data_full = backward(img, dim)
    else:
        data_full = np.copy(img)

    acs = np.copy(data_full)

    # Prepare slices for selecting ACS lines
    lines = [slice(None) for l in range(len(data_full.shape))]
    
    # Iterate over acceleration factors and modify data_full
    for i, j in enumerate(af):
        shp = data_full.shape[dim[i]]
        if j > 1:
            lines[dim[i]] = slice(shp//2 - nb_lines//2 + acs_shift[i],
                                  shp//2 + nb_lines//2 + acs_shift[i])
        for k in range(j)[:-1]:
            data_full[tuple(
                slice(k, None, j) if l == dim[i] else slice(None)
                for l in range(len(data_full.shape))
                )] = 0
    
    # Extract ACS lines from the full data
    acs = acs[tuple(lines)]
    
    return data_full, acs


class B0simu:
    def __init__(self, rawk, fmap, vsize):
        self.fullk = rawk
        self.fulli = forward(rawk, (0, 1, 3))
        self.recoi = np.linalg.norm(self.fulli, axis=2)
        self.fmap = fmap
        self.gmap = np.gradient(self.fmap)
        if type(vsize) == float:
            vsize = [vsize] * 3
        self.vsize = vsize
        self.resk = np.zeros(self.fullk.shape, dtype=np.complex64)

    def performSimu(self, TE=20e-3, t_obs=1e-3, method='very_simple', upsample_factor=3):
        ky, kz, _, kx = self.fullk.shape
        self.k = [ky, kz, kx]
        t_i = np.linspace(-0.5, 0.5, kx)
        t = TE + t_i * t_obs
        self.TE = TE
        self.t_obs = t_obs
        self.t = t
        tx, ty, tz = (np.arange(t) for t in [kx, ky, kz])
        gtx = 2 * pi * (tx - kx//2) / self.vsize[2] / kx
        gty = 2 * pi * (ty - ky//2) / self.vsize[0] / ky
        gtz = 2 * pi * (tz - kz//2) / self.vsize[1] / kz
        GTy, GTz, GTx = np.meshgrid(gty, gtz, gtx, indexing='ij')
        GTx_use = GTx * self.vsize[2] / 2 / pi
        GTy_use = GTy * self.vsize[0] / 2 / pi
        GTz_use = GTz * self.vsize[1] / 2 / pi
        self.gt = [gty, gtz, gtx]
        self.GT = [GTy, GTz, GTx]
        self.GT_use = [GTy_use, GTz_use, GTx_use]

        # TODO ignore where fmap == 0 and gmap == 0
        iss = np.stack(np.where(self.fmap != 1000)).T
        self._apply_dephasing(iss, method=method, upsample_factor=upsample_factor)
        

    def _apply_dephasing(self, iss, method='simple', upsample_factor=3):
        if method == 'full':
            tf = torch.from_numpy(self.t)
            res_t = torch.from_numpy(self.resk)
            g = torch.from_numpy(self.fmap)
            gm = torch.gradient(g)
            ffulli = torch.from_numpy(self.fulli)
            gtf = []
            gtusef = []
            for i in range(3):
                gtf.append(torch.from_numpy(self.gt[i]))
                gtusef.append(torch.from_numpy(self.GT_use[i]))
            for i in tqdm(iss):
                factor = (torch.sinc((gtf[2] * self.vsize[2]/2 + 2 * pi * gm[2][*i] / 2 * tf) / pi)[None, None, :]
                          * torch.sinc(
                              (
                                  gtusef[0] + (2 * pi * gm[0][*i] * tf / 2 / pi)[None, None, :]
                              ))
                          * torch.sinc(
                              (
                                  gtusef[1] + (2 * pi * gm[1][*i] * tf / 2 / pi)[None, None, :]
                              ))
                          * (torch.exp(2 * pi * 1j * g[*i] * tf))[None, None, :])
                slic = (i[0], i[1], slice(None), i[2])
                res_t += (
                        (( torch.exp(-1j * gtf[2] * (i[2] - self.k[2]//2) * self.vsize[2]))[None, None, :]
                        * (torch.exp(-1j * gtf[0] * (i[0] - self.k[0]//2) * self.vsize[0]))[:, None, None]
                        * (torch.exp(-1j * gtf[1] * (i[1] - self.k[1]//2) * self.vsize[1]))[None, :, None]
                        * factor)[:, :, None, :]
                        ) * ffulli[slic][None, None, :, None]
            self.resk = res_t.numpy()
        elif method == 'simple':
            for i in tqdm(iss):
                factor = ((np.sinc((self.gt[2] * self.vsize[2]/2 + 2 * pi * self.gmap[2][*i] / 2 * self.t) / pi) / np.sinc(self.gt[2] * self.vsize[2]/2 / pi))
                          * np.sinc(2 * pi * self.gmap[0][*i] / 2 * self.t / pi)
                          * np.sinc(2 * pi * self.gmap[1][*i] / 2 * self.t / pi)
                          * (np.exp(2 * pi * -1j * self.fmap[*i] * self.t)))
                slic = (i[0], i[1], slice(None), i[2])
                self.resk[i[0], i[1], :, :] += (
                        (( np.exp(-1j * self.gt[2] * (i[2] - self.k[2]//2) * self.vsize[2]))
                        * factor)[None, :]
                        ) * self.fulli[slic][:, None]
            self.resk = backward(self.resk, (0, 1))
        elif method == 'very_simple':
            factor = (np.sinc(2 * pi * self.gmap[2] / 2 * self.TE / pi)
                      * np.sinc(2 * pi * self.gmap[0] / 2 * self.TE / pi)
                      * np.sinc(2 * pi * self.gmap[1] / 2 * self.TE / pi)
                      * (np.exp(2 * pi * 1j * self.fmap * self.TE)))
            self.resk = self.fulli * factor[:, :, None, :]
            self.resk = backward(self.resk, (0, 1, 3))
        elif method == 'upsample':
            print("upsampling... this may take a lot of memory.")
            factor = upsample_factor
            big_t = np.linspace(-0.5, 0.5, self.fulli.shape[3]*factor) * self.t_obs + self.TE
            # big_arr = self.fulli.repeat(factor, axis=0).repeat(factor, axis=1).repeat(factor, axis=3)
            big_arr = self.fulli.repeat(factor, axis=1).repeat(factor, axis=3)
            points = [np.linspace(0, 1, self.fullk.shape[i]) for i in [0, 1, 3]]
            points_up = np.meshgrid(*[np.linspace(0, 1, self.fullk.shape[i]*f) for i,f in zip([0, 1, 3], [1, factor, factor])], indexing='ij')
            mapper = interpolate.interpn(points, self.fmap, points_up, method='linear')
            if self.fmap.max() == 0:
                big_fmap = np.zeros_like(mapper)
            else:
                big_fmap = (mapper / mapper.max()) * self.fmap.max()
            # big_arr *= np.exp(2 * pi * 1j * big_fmap * self.TE)[:, :, None, :]
            c = np.array([*big_arr.shape]) // 2
            r = c + np.array([*self.fulli.shape]) // 2
            l = c - np.array([*self.fulli.shape]) // 2
            big_k = np.zeros_like(big_arr)
            for i in tqdm(range(big_arr.shape[3])):
                bn = np.zeros_like(big_arr)
                bn[:, :, :, i] = big_arr[:, :, :, i]
                big_k += backward(bn, (3,)) * np.exp(2 * pi * 1j * big_fmap[:,:,i][:,:,None] * big_t[None, None, :])[:,:,None,:]

            self.resk = backward(big_k, (0, 1))[l[0]:r[0], l[1]:r[1], :, l[3]:r[3]]


    def performGrappa(self, af, nb_lines):
        data_full, acs = extract_data_acs(self.resk, [0, 1], af=[2,2], nb_lines=24)
        kernel_size = (4, 4, 5)
        lambda_ = 1e-4

        cuda = False
        cuda_mode = 'all'

        data_t = torch.from_numpy(data_full).permute((2, 1, 0, 3))
        acs_t = torch.from_numpy(acs).permute((2, 1, 0, 3))
        sig, grappa_recon_spec = GRAPPA_Recon(data_t,
                                              acs_t,
                                              af=af,
                                              delta=0,
                                              grappa_recon_spec=None,
                                              quiet=True,
                                              kernel_size=kernel_size,
                                              lambda_=lambda_,
                                              cuda=cuda,
                                              cuda_mode=cuda_mode)
        self.sig = sig.permute((2, 1, 0, 3)).cpu().numpy()




class B0simu2D:
    def __init__(self, rawk, fmap, inplane_gradient, vsize):
        self.fullk = rawk
        self.fulli = forward(rawk, (0, 2))
        self.recoi = np.linalg.norm(self.fulli, axis=1)
        self.fmap = fmap
        self.gmap = np.gradient(self.fmap)
        self.inplane_gradient = inplane_gradient
        if type(vsize) == float:
            vsize = [vsize] * 3
        self.vsize = vsize
        self.resk = np.zeros(self.fullk.shape, dtype=np.complex64)

    def performSimu(self, TE=20e-3, t_obs=1e-3, method='very_simple', upsample_factor=3, max_jobs=8):
        kz, _, kx = self.fullk.shape
        self.k = [kz, kx]
        t_i = np.linspace(-0.5, 0.5, kx)
        t = TE + t_i * t_obs
        self.TE = TE
        self.t_obs = t_obs
        self.t = t
        tx, tz = (np.arange(t) for t in [kx, kz])
        gtx = 2 * pi * (tx - kx//2) / self.vsize[2] / kx
        gtz = 2 * pi * (tz - kz//2) / self.vsize[1] / kz
        GTz, GTx = np.meshgrid(gtz, gtx, indexing='ij')
        GTx_use = GTx * self.vsize[2] / 2 / pi
        GTz_use = GTz * self.vsize[1] / 2 / pi
        self.gt = [gtz, gtx]
        self.GT = [GTz, GTx]
        self.GT_use = [GTz_use, GTx_use]

        if method == 'very_simple':
            factor = (np.sinc(2 * pi * self.gmap[1] / 2 * self.TE / pi)
                      * np.sinc(2 * pi * self.gmap[0] / 2 * self.TE / pi)
                      * np.sinc(2 * pi * self.inplane_gradient[1] / 2 * self.TE / pi)
                      * (np.exp(2 * pi * -1j * self.fmap * self.TE)))
            self.resk = self.fulli * factor[:, None, :]
            self.resk = backward(self.resk, (0, 2))
            return

        # TODO ignore where fmap == 0 and gmap == 0
        iss = np.stack(np.where(self.fmap != 1000)).T
        # self._apply_dephasing(iss, method=method, upsample_factor=upsample_factor)
        splits = 200
        a = thread_map(self._apply_dephasing, np.array_split(iss, splits), [method]*splits, [upsample_factor]*splits, max_workers=max_jobs)
        self.resk = np.sum(a, axis=0)

    def _apply_dephasing(self, iss, method='simple', upsample_factor=3):
        if method == 'full':
            re = np.zeros_like(self.resk)
            for i in iss:
                factor = (np.sinc((self.gt[1] * self.vsize[2]/2 + 2 * pi * self.gmap[1][*i] / 2 * self.t) / pi)[None, :]
                          * np.sinc(
                              (
                                  self.GT_use[0] + (2 * pi * self.gmap[0][*i] * self.t / 2 / pi)[None, :]
                              ))
                          * np.sinc(
                              (
                                  self.GT_use[1] + (2 * pi * self.inplane_gradient[*i] * self.t / 2 / pi)[None, :]
                              ))
                          * (np.exp(2 * pi * -1j * self.fmap[*i] * self.t))[None, :])
                slic = (i[0], slice(None), i[1])
                re += (
                        ((np.exp(-1j * self.gt[0] * (i[0] - self.k[0]//2) * self.vsize[0]))[:, None]
                        * (np.exp(-1j * self.gt[1] * (i[1] - self.k[1]//2) * self.vsize[1]))[None, :]
                        * factor)[:, None, :]
                        ) * self.fulli[slic][None, :, None]

            return re
