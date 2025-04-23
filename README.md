# MRI_utils
my tools.

## examples

### plot ipe

```py
# overlay B0 map on anat image

import numpy as np
import matplotlib
from MRI_utils.plot import save_quadrants_ipe

img = np.load('img.npy') # anat
B0  = np.load('B0.npy')  # B0 map
# assuming both are the same shape

pos = np.unravel_index(np.argmax(B0), B0.shape)
cmap = matplotlib.pyplot.get_cmap('jet')
mcmap = cmap(np.arange(cmap.N))
mcmap[50:, -1] = 0.8
mcmap[:50, -1] = np.linspace(0, 0.8, 50) # low values are very transparent
cmap = matplotlib.colors.ListedColormap(mcmap)

save_quadrants_ipe([B0, img], pos, cmaps=[cmap, 'gray'])
```

![result](/ressources/B0_overlayed.svg)
