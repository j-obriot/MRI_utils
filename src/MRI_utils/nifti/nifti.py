import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.orientations import (
    io_orientation,
    axcodes2ornt,
    ornt_transform,
    apply_orientation,
    inv_ornt_aff
)
from .slicer import SliceViewer
import imageio
from pathlib import Path

def _prepare_data(data, affine):
    orig_ornt = io_orientation(affine)
    ras_ornt = axcodes2ornt(('R', 'A', 'S'))

    transform = ornt_transform(orig_ornt, ras_ornt)
    data_ras = apply_orientation(data, transform)

    inv_affine_ornt = inv_ornt_aff(transform, data.shape)

    return data_ras, inv_affine_ornt

def _load_and_prepare(img):
    """
    Returns:
        data_ras, affine, inv_ornt_affine
    """

    if isinstance(img, str):
        nii = nib.load(img)
        if len(nii.shape) > 3:
            data = nii.dataobj[..., 0]
        else:
            data = nii.get_fdata()
        affine = nii.affine
        data_ras, inv_aff = _prepare_data(data, affine)

    elif isinstance(img, nib.Nifti1Image):
        if len(img.shape) > 3:
            data = img.dataobj[..., 0]
        else:
            data = img.get_fdata()
        affine = img.affine
        data_ras, inv_aff = _prepare_data(data, affine)

    elif isinstance(img, np.ndarray):
        data_ras = img
        if data_ras.ndim > 3:
            data_ras = data_ras[..., 0]
        affine = np.eye(4)
        inv_aff = np.eye(4)

    else:
        raise ValueError("Unsupported input type")

    return data_ras, affine, inv_aff

def interactive_slices(img, **imshow_kwargs):
    """
    img : str | nib.Nifti1Image | np.ndarray
    **imshow_kwargs : forwarded to matplotlib.imshow
    """

    data_ras, affine, inv_aff = _load_and_prepare(img)

    viewer = SliceViewer(data_ras, affine, inv_aff, **imshow_kwargs)
    return viewer.show()


def _extract_slice(data_ras, coords, view):
    if view == 'sag':
        return data_ras[coords['sag'], :, :]
    elif view == 'cor':
        return data_ras[:, coords['cor'], :]
    elif view == 'ax':
        return data_ras[:, :, coords['ax']]
    else:
        raise ValueError("view must be 'sag', 'cor', or 'ax'")


def save_selected_slice(
    img,
    view,
    filename,
    coords=None,
    **imshow_kwargs
):
    """
    Parameters
    ----------
    img : input image
    view : 'sag' | 'cor' | 'ax'
    filename : output image path
    coords : dict or None
        If None → interactive selection
        Else → {'voxel_ras': [x,y,z], ...}
    **imshow_kwargs : passed to imshow

    Returns
    -------
    dict (same as interactive_slices)
    """

    data_ras, affine, inv_aff = _load_and_prepare(img)

    if coords is None:
        coords = interactive_slices(img, **imshow_kwargs)

    # normalize coords format
    if 'voxel_ras' in coords:
        idx = {
            'sag': int(coords['voxel_ras'][0]),
            'cor': int(coords['voxel_ras'][1]),
            'ax': int(coords['voxel_ras'][2])
        }
    else:
        idx = coords

    slc = _extract_slice(data_ras, idx, view)

    # default saving parameters
    vmin, vmax = np.quantile(data_ras, [0.01, 0.99])
    imkw = {"cmap": "gray",
            "vmin": vmin, # ignore 1% low and high
            "vmax": vmax, # for scale.
            **imshow_kwargs}

    plt.imsave(filename, np.rot90(slc), **imkw)

    return coords


def create_slice_gif(img,
                     orientation='ax',
                     output_file='slices.gif',
                     duration=0.1,
                     vmin=None, vmax=None,
                     cmap='gray'):
    """
    Create a GIF of slices with minimal loss.
    """

    ext = Path(output_file).suffix.lower()
    if ext in ['.gif', '.apng', '.mp4']:
        format_ = ext[1:]
    else:
        raise ValueError("Could not infer format from extension. Use .gif, .apng, or .mp4")


    def _prepare_data(data, affine):
        orig_ornt = io_orientation(affine)
        ras_ornt = axcodes2ornt(('R', 'A', 'S'))
        transform = ornt_transform(orig_ornt, ras_ornt)
        data_ras = apply_orientation(data, transform)
        if data_ras.ndim > 3:
            data_ras = data_ras[..., 0]
        return data_ras

    if isinstance(img, str):
        nii = nib.load(img)
        data = nii.get_fdata()
        affine = nii.affine
        data_ras = _prepare_data(data, affine)
    elif isinstance(img, nib.Nifti1Image):
        data = img.get_fdata()
        affine = img.affine
        data_ras = _prepare_data(data, affine)
    elif isinstance(img, np.ndarray):
        data_ras = img
    else:
        raise ValueError("Unsupported input type")

    axis = {'sag':0, 'cor':1, 'ax':2}[orientation]

    # Normalize slices to 0-255 for minimal loss
    if vmin is None:
        vmin = np.quantile(data_ras, 0.01)
    if vmax is None:
        vmax = np.quantile(data_ras, 0.99)

    def normalize(slice_):
        slice_ = np.clip(slice_, vmin, vmax)
        slice_ = (slice_ - vmin) / (vmax - vmin)
        return (slice_ * 255).astype(np.uint8)

    # Build frames
    frames = []
    for i in range(data_ras.shape[axis]):
        if axis == 0:
            slc = data_ras[i, :, :]
        elif axis == 1:
            slc = data_ras[:, data_ras.shape[1] - 1 - i, :]
        else:
            slc = data_ras[:, :, data_ras.shape[2] - 1 - i]

        slc_norm = normalize(np.rot90(slc))
        frames.append(slc_norm)

    if format_ == 'gif':
        imageio.mimsave(output_file, frames, duration=duration, mode='L')
    elif format_ == 'apng':
        imageio.mimsave(output_file, frames, duration=duration, format='APNG')
    elif format_ == 'mp4':
        fps = int(1/duration)
        imageio.mimsave(output_file, frames, fps=fps, codec='libx264', quality=8)


def create_epi_gif(img,
                   orientation='ax',
                   output_file='epi.gif',
                   coords=None,
                   duration=0.1,
                   vmin=None, vmax=None,
                   cmap='gray'):
    """
    Create a GIF of slices with minimal loss.
    """

    ext = Path(output_file).suffix.lower()
    if ext in ['.gif', '.apng', '.mp4']:
        format_ = ext[1:]
    else:
        raise ValueError("Could not infer format from extension. Use .gif, .apng, or .mp4")

    if coords is None:
        coords = interactive_slices(img)

    # normalize coords format
    if 'voxel_ras' in coords:
        idx = {
            'sag': int(coords['voxel_ras'][0]),
            'cor': int(coords['voxel_ras'][1]),
            'ax': int(coords['voxel_ras'][2])
        }
    else:
        idx = coords


    def _prepare_data(data, affine):
        orig_ornt = io_orientation(affine)
        ras_ornt = axcodes2ornt(('R', 'A', 'S'))
        transform = ornt_transform(orig_ornt, ras_ornt)
        data_ras = apply_orientation(data, transform)
        return data_ras

    if isinstance(img, str):
        nii = nib.load(img)
        data = nii.get_fdata()
        affine = nii.affine
        data_ras = _prepare_data(data, affine)
    elif isinstance(img, nib.Nifti1Image):
        data = img.get_fdata()
        affine = img.affine
        data_ras = _prepare_data(data, affine)
    elif isinstance(img, np.ndarray):
        data_ras = img
    else:
        raise ValueError("Unsupported input type")

    axis = {'sag':0, 'cor':1, 'ax':2}[orientation]

    # Normalize slices to 0-255 for minimal loss
    if vmin is None:
        vmin = np.quantile(data_ras, 0.01)
    if vmax is None:
        vmax = np.quantile(data_ras, 0.99)

    def normalize(slice_):
        slice_ = np.clip(slice_, vmin, vmax)
        slice_ = (slice_ - vmin) / (vmax - vmin)
        return (slice_ * 255).astype(np.uint8)

    # Build frames
    frames = []
    for i in range(data_ras.shape[3]):
        if axis == 0:
            slc = data_ras[idx[orientation], :, :, i]
        elif axis == 1:
            slc = data_ras[:, idx[orientation], :, i]
        else:
            slc = data_ras[:, :, idx[orientation], i]

        slc_norm = normalize(np.rot90(slc))
        frames.append(slc_norm)

    if format_ == 'gif':
        imageio.mimsave(output_file, frames, duration=duration, mode='L')
    elif format_ == 'apng':
        imageio.mimsave(output_file, frames, duration=duration, format='APNG')
    elif format_ == 'mp4':
        fps = int(1/duration)
        imageio.mimsave(output_file, frames, fps=fps, codec='libx264', quality=8)

    return coords
