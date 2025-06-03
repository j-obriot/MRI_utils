import numpy as np
import nibabel as nib
import os
import glob
from pathlib import Path

class DataSet:
    def __init__(self, init=None):
        envdata = os.environ.get("DATASET")
        if init is not None:
            self.dir = Path(init)
        elif envdata is not None:
            self.dir = Path(envdata)
        else:
            self.dir = Path("./")
        self.dir = self.dir.expanduser().resolve()
        if not self.dir.is_dir():
            raise FileNotFoundError("No such directory")

    def cd(self, new_dir):
        ndir = (self.dir / new_dir).resolve()
        print(ndir)
        if not ndir.is_dir() :
            raise FileNotFoundError("No such directory")
        self.dir = ndir
    def ls(self):
        for f in sorted(glob.glob(self.dir / "*")):
            print(f[len(str(self.dir)):])

    def get_file(self, filename):
        file = (self.dir / filename).resolve()
        if not file.is_file():
            raise FileNotFoundError("No such file")
        return file

    def glob(self, pattern):
        return glob.glob(str(self.dir / pattern))

    def load_file(self, filename):
        file = str(self.get_file(filename))
        accepted_ext = ["nii.gz", "nii", "npy"]
        resolver = [nib.load, nib.load, np.load]
        for i, e in enumerate(accepted_ext):
            if file[-len(e)-1:] == f".{e}":
                return resolver[i](file)
        raise TypeError("unsupported type")

    def load_arr(self, filename):
        file = self.load_file(filename)
        if type(file) == nib.Nifti1Image:
            return file.get_fdata()
        return file

    def __repr__(self):
        return str(self.dir)
    def __str__(self):
        return str(self.dir)
