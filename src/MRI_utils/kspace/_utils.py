import numpy as np

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

