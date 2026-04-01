import numpy as np
import matplotlib.pyplot as plt

class SliceViewer:
    def __init__(self, data_ras, affine, inv_ornt_affine, **imshow_kwargs):
        self.data = data_ras
        self.affine = affine
        self.inv_ornt_affine = inv_ornt_affine
        self.imshow_kwargs = imshow_kwargs

        if self.data.ndim > 3:
            self.data = self.data[..., 0]

        self.shape = self.data.shape

        self.idx = {
            'sag': self.shape[0] // 2,
            'cor': self.shape[1] // 2,
            'ax': self.shape[2] // 2
        }

        self.vmin = np.quantile(self.data, 0.01)
        self.vmax = np.quantile(self.data, 0.99)

        self.fig, self.axes = plt.subplots(1, 3)
        self._connect()
        self._draw()

    def _get_slices(self):
        return (
            self.data[self.idx['sag'], :, :],
            self.data[:, self.idx['cor'], :],
            self.data[:, :, self.idx['ax']]
        )

    def _draw_cross(self, ax, x, y):
        ax.axhline(y, color='r', lw=0.8)
        ax.axvline(x, color='r', lw=0.8)

    def _draw(self):
        for a in self.axes:
            a.clear()

        sag, cor, ax = self._get_slices()

        imkw = {"cmap": "gray",
                "vmin": self.vmin,
                "vmax": self.vmax,
                "interpolation": "nearest",
                **self.imshow_kwargs}

        # Sagittal
        self.axes[0].imshow(np.rot90(sag), **imkw)
        self.axes[0].set_title(f"Sagittal x={self.idx['sag']}")
        self._draw_cross(self.axes[0],
                         self.idx['cor'],
                         self.shape[2] - self.idx['ax'])

        # Coronal
        self.axes[1].imshow(np.rot90(cor), **imkw)
        self.axes[1].set_title(f"Coronal y={self.idx['cor']}")
        self._draw_cross(self.axes[1],
                         self.idx['sag'],
                         self.shape[2] - self.idx['ax'])

        # Axial
        self.axes[2].imshow(np.rot90(ax), **imkw)
        self.axes[2].set_title(f"Axial z={self.idx['ax']}")
        self._draw_cross(self.axes[2],
                         self.idx['sag'],
                         self.shape[1] - self.idx['cor'])

        for a in self.axes:
            a.axis('off')

        self.fig.canvas.draw_idle()

    def _onclick(self, event):
        if event.inaxes not in self.axes:
            return

        ax_id = self.axes.tolist().index(event.inaxes)
        x, y = int(event.xdata), int(event.ydata)

        if ax_id == 0:
            self.idx['cor'] = np.clip(x, 0, self.shape[1] - 1)
            self.idx['ax'] = np.clip(self.shape[2] - y, 0, self.shape[2] - 1)

        elif ax_id == 1:
            self.idx['sag'] = np.clip(x, 0, self.shape[0] - 1)
            self.idx['ax'] = np.clip(self.shape[2] - y, 0, self.shape[2] - 1)

        elif ax_id == 2:
            self.idx['sag'] = np.clip(x, 0, self.shape[0] - 1)
            self.idx['cor'] = np.clip(self.shape[1] - y, 0, self.shape[1] - 1)

        self._draw()

    def _on_scroll(self, event):
        if event.inaxes not in self.axes:
            return
        ax_id = self.axes.tolist().index(event.inaxes)
        delta = int(np.sign(event.step))  # scroll direction

        if ax_id == 0:
            self.idx['sag'] = np.clip(self.idx['sag'] + delta, 0, self.shape[0]-1)
        elif ax_id == 1:
            self.idx['cor'] = np.clip(self.idx['cor'] + delta, 0, self.shape[1]-1)
        elif ax_id == 2:
            self.idx['ax'] = np.clip(self.idx['ax'] + delta, 0, self.shape[2]-1)

        self._draw()

    def _connect(self):
        self.fig.canvas.mpl_connect('button_press_event', self._onclick)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)

    def _compute_outputs(self):
        voxel_ras = np.array([
            self.idx['sag'],
            self.idx['cor'],
            self.idx['ax'],
            1
        ])

        voxel_native = (self.inv_ornt_affine @ voxel_ras)[:3]
        world = (self.affine @ np.append(voxel_native, 1))[:3]

        return {
            'voxel_ras': voxel_ras[:3].astype(int),
            'voxel_native': voxel_native.astype(float),
            'world_mm': world.astype(float)
        }

    def show(self):
        plt.show()
        return self._compute_outputs()
