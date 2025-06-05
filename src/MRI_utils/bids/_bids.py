import os
import glob
from ..dataset import DataSet

def get_function_session(directory):
    if type(directory) == DataSet:
        directory = str(directory)
    subjects_dirs = sorted(glob.glob(os.path.join(directory, 'sub-*/')))

    sessions = {}

    for i in subjects_dirs:
        func = os.path.join(i, "func")
        funcs = glob.glob(os.path.join(i, "*", "func"))
        sub = i[len(directory)+5:-1]

        if os.path.isdir(func):
            if "" not in sessions.keys():
                sessions[""] = {}
            sessions[""][sub] = func
        else:
            for ses in funcs:
                sesname = ses[len(directory)+5+len(sub)+5:-5]
                if os.path.isdir(ses):
                    if sesname not in sessions.keys():
                        sessions[sesname] = {}
                    sessions[sesname][sub] = ses
    return sessions

def _get_attribute(filename, attribute):
    # does not support attributes with dot
    attrs = filename.split('.')[0].split('_')
    for a in attrs:
        if a[:len(attribute)+1] == attribute + '-':
            return a[len(attribute)+1:]
    return None

def _get_lone_attribute(filename, attribute):
    # does not support attributes with dot
    attrs = filename.split('.')[0].split('_')
    for a in attrs:
        if a == attribute:
            return True
    return False

def get_attr_files(directory, keys, basename=False):
    """
    returns the files in a structure {session: {subject: [file1, file2]}}

    Keys are constraints in a structure to indicate which files to include.
    keys can be prepended with:
    '*' to indicate a lone attribute and
    '!' to invert the result
    values of keys can be cumulated with '|'
    {
        "space": "T1w",              # match files that are in space T1w
        "extension": "nii.gz|gii",   # match files with extension nii.gz or gii
        "*bold": "",                 # match the bold lone attribute
        "!task": "fpptrn"            # match anything but files with this task
    }
    this structure would match this file for example:
    sub-h619131_task-fppval_run-01_space-T1w_bold.nii.gz
    """
    sessions = get_function_session(directory)

    for sesname, session in sessions.items():
        for subject, func_dir in session.items():
            files = [
                    os.path.basename(f)
                    for f in glob.glob(os.path.join(func_dir, '*'))
                    ]
            acc_files = []
            for f in files:
                cont = False
                for k, a in keys.items():
                    invert = False
                    if k[0] == '!':
                        invert = True
                        k = k[1:]
                    if k == "extension":
                        temp = False
                        for g in a.split('|'):
                            if f[-len(g):] == g:
                                temp = True
                        if (not temp) != invert:
                            cont = True
                            break
                    elif k[0] == "*":
                        if (not _get_lone_attribute(f, k[1:])) != invert:
                            cont = True
                            break
                    elif (_get_attribute(f, k) not in a.split('|')) != invert:
                        cont = True
                        break
                if cont:
                    continue
                fpath = os.path.join(func_dir, f)
                acc_files.append(f if basename else fpath)
            sessions[sesname][subject] = sorted(acc_files)
    return sessions
