import numpy as np
import base64
import io
import matplotlib.pyplot as plt
import matplotlib

def save_quadrants_ipe(filename, arrs, pos=[], psi=0, theta=30, phi=30, cmaps=[], norms=[]):
    """
    Save 3D array slices as quadrants projected in a visualization plane in an IPE XML file with specified rotations and color maps.

    Parameters:
    - filename (str): The name of the output IPE file.
    - arrs (list or np.ndarray): A list of 3D arrays or a single 3D array to be sliced and saved.
    - pos (list, optional): The slicing position in the format [x, y, z]. Defaults to the center of the array.
    - psi (float, optional): Rotation angle around the x-axis in degrees. Defaults to 0.
    - theta (float, optional): Rotation angle around the y-axis in degrees. Defaults to 30.
    - phi (float, optional): Rotation angle around the z-axis in degrees. Defaults to 30.
    - cmaps (list, optional): List of colormaps for each array. Defaults to 'gray' for all.
    - norms (list, optional): List of normalization objects for each array. Defaults to automatic normalization.

    Returns:
    None
    """
    psi = np.radians(psi)
    theta = np.radians(theta)
    phi = np.radians(phi)

    Rx = np.array([[1, 0,            0],
                   [0, np.cos(psi), -np.sin(psi)],
                   [0, np.sin(psi),  np.cos(psi)]])
    Ry = np.array([[np.cos(theta),  0, np.sin(theta)],
                   [0,              1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0],
                   [np.sin(phi),  np.cos(phi), 0],
                   [0,            0,           1]])

    M = Rx @ Ry @ Rz

    TR = np.eye(3)
    TR[:2, 2] = 400
    TRm = np.eye(3)
    TRm[:2, 2] = -400
    XY = np.eye(3)
    XY[:2, :2] = M[1:, :2]
    XYm = TR @ XY @ TRm
    XZ = np.eye(3)
    XZ[:2, :2] = M[1:, [0, 2]]
    XZm = TR @ XZ @ TRm
    YZ = np.eye(3)
    YZ[:2, :2] = M[1:, 1:]
    YZm = TR @ YZ @ TRm

    xy = list(XYm[:2, :].T.flatten())
    xz = list(XZm[:2, :].T.flatten())
    yz = list(YZm[:2, :].T.flatten())

    def slicer(arr, pos, center, mats, cmaps = [], norms = []):
        arr = [a[:, :, ::-1] for a in arr]
        if len(cmaps) != len(arr):
            cmaps = ['gray'] * len(arr)
            cmap = [plt.get_cmap(i) for i in cmaps]
        else:
            cmap = [plt.get_cmap(i) if type(i) == str else i for i in cmaps]
        if len(norms) != len(arr):
            norm = [matplotlib.colors.Normalize(
                vmin=np.nanmin(arr[i]),
                vmax=np.nanmax(arr[i])) for i in range(len(arr))]
        else:
            norm = norms

        # assuming the matrices are the same shape
        sh = arr[0].shape

        ixys = [np.rot90(255 * cmap[i](norm[i](arr[i][pos[0], :, :]))).astype(np.uint8) for i in range(len(arr))]
        pxy = (center - pos[1], center - (sh[2] - pos[2]), center + (sh[1] - pos[1]), center + pos[2])

        ixzs = [(255 * cmap[i](norm[i](arr[i][:, :, (sh[2] - pos[2])]))).astype(np.uint8) for i in range(len(arr))]
        pxz = (center - pos[1], center - (sh[0] - pos[0]), center + (sh[1] - pos[1]), center + pos[0])

        iyzs = [(255 * cmap[i](norm[i](arr[i][:, pos[1], :]))).astype(np.uint8) for i in range(len(arr))]
        pyz = (center - (sh[2] - pos[2]), center - (sh[0] - pos[0]), center + pos[2], center + pos[0])

        pos2y2 = sh[2] - pos[2]
        toret = []

        # PTDR
        # foreground
        #     |
        #     v
        # background
        # works for angles < 90
        # TODO: change order depending on angles.
        ims = [
                *[{'i': ixy[pos[2]:, pos[1]:],
                   'p': (center, pxy[1], pxy[2], center),
                   'm': mats[0]} for ixy in ixys],
                *[{'i': ixz[:pos[0], pos[1]:],
                   'p': (center, center, pxz[2], pxz[3]),
                   'm': mats[1]} for ixz in ixzs],
                *[{'i': iyz[:pos[0], :pos2y2],
                   'p': (pyz[0], center, center, pyz[3]),
                   'm': mats[2]} for iyz in iyzs],

                *[{'i': ixy[:pos[2], pos[1]:],
                   'p': (center, center, pxy[2], pxy[3]),
                   'm': mats[0]} for ixy in ixys],
                *[{'i': ixz[pos[0]:, pos[1]:],
                   'p': (center, pxz[1], pxz[2], center),
                   'm': mats[1]} for ixz in ixzs],
                *[{'i': iyz[:pos[0], pos2y2:],
                   'p': (center, center, pyz[2], pyz[3]),
                   'm': mats[2]} for iyz in iyzs],

                *[{'i': ixy[pos[2]:, :pos[1]],
                   'p': (pxy[0], pxy[1], center, center),
                   'm': mats[0]} for ixy in ixys],
                *[{'i': ixz[:pos[0], :pos[1]],
                   'p': (pxz[0], center, center, pxz[3]),
                   'm': mats[1]} for ixz in ixzs],
                *[{'i': iyz[pos[0]:, :pos2y2],
                   'p': (pyz[0], pyz[1], center, center),
                   'm': mats[2]} for iyz in iyzs],

                *[{'i': ixy[:pos[2], :pos[1]],
                   'p': (pxy[0], center, center, pxy[3]),
                   'm': mats[0]} for ixy in ixys],
                *[{'i': ixz[pos[0]:, :pos[1]],
                   'p': (pxz[0], pxz[1], center, center),
                   'm': mats[1]} for ixz in ixzs],
                *[{'i': iyz[pos[0]:, pos2y2:],
                   'p': (center, pyz[1], pyz[2], center),
                   'm': mats[2]} for iyz in iyzs],
                ][::-1]

        return ims

    center = 400

    arrs = np.array(arrs)
    if len(arrs.shape) == 3:
        arrs = [arrs]
    elif len(arrs.shape) != 4:
        print("please provide a 3D of a list of 3D arrays to plot.")
        return
    if len(pos) != 3:
        sh = arrs[0].shape
        pos = (sh[0]//2, sh[1]//2, sh[2]//2)

    ims = slicer(arrs, pos, center, [xy, xz, yz], cmaps=cmaps, norms=norms)


    with open(filename, 'w') as txt:
        txt.write(f"""<?xml version=\"1.0\"?>
    <!DOCTYPE ipe SYSTEM \"ipe.dtd\">
    <ipe version=\"70218\" creator=\"Ipe 7.2.28\">
    <info created=\"D:20250407172152\" modified=\"D:20250407172152\"/>
    """)
        for j, im in enumerate(ims):
            slices = im['i']
            if np.array_equal(slices[:, :, 0], slices[:, :, 1]) and np.array_equal(slices[:, :, 1], slices[:, :, 2]):
                isgray = True
                c = slices[:, :, 0]
            else:
                isgray = False
                c = slices[:, :, :3]
            a = slices[:, :, 3]
            txt.write(f"""<bitmap id=\"{j}\" width=\"{slices.shape[1]}\" height=\"{slices.shape[0]}\" BitsPerComponent=\"8\" ColorSpace=\"Device{'Gray' if isgray else 'RGB'}Alpha\" length=\"{slices.shape[0] * slices.shape[1] * 3}\" alphaLength=\"{slices.shape[0] * slices.shape[1]}\" encoding=\"base64\">
    """)

            txt.write(base64.b64encode(c.tobytes() + a.tobytes()).decode())

            txt.write(f"""
    </bitmap>
    """)
        txt.write(f"""<page>
    <layer name=\"alpha\"/>
    <view layers=\"alpha\" active=\"alpha\"/>
    """)
        for j, i in enumerate(ims):
            p = i['p']
            m = i['m']
            txt.write(f"""<image layer=\"alpha\" matrix=\"{' '.join([f"{i:0.5f}" for i in m])}\" rect=\"{' '.join([f"{i}" for i in p])}\" bitmap=\"{j}\"/>
    """)
        txt.write("""</page>
    </ipe>""")

# TODO save_quadrants_svg
# xml is very similar so could be a straightforward implementation
