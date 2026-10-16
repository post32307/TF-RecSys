import world
import numpy as np
from dataloader import BasicDataset
from time import time
import glob
import importlib.util
import os
import sysconfig

try:
    sources_path = os.path.join(os.path.dirname(__file__), "sources")
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    ext_candidates = glob.glob(os.path.join(sources_path, f"sampling*{ext_suffix}"))
    if ext_candidates:
        spec = importlib.util.spec_from_file_location("sampling", ext_candidates[0])
        sampling = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sampling)
    else:
        from cppimport import imp_from_filepath
        path = os.path.join(sources_path, "sampling.cpp")
        sampling = imp_from_filepath(path)
    sampling.seed(world.seed)
    sample_ext = True
except BaseException as e:
    world.cprint(f"Cpp extension not loaded: {e}")
    sample_ext = False


def UniformSample_original(dataset, neg_ratio = 1):
    dataset : BasicDataset
    allPos = dataset.allPos
    start = time()
    if sample_ext:
        S = sampling.sample_negative(dataset.n_users, dataset.m_items,
                                     dataset.trainDataSize, allPos, neg_ratio)
        S = np.array([np.array(x) for x in S if (0<=x[0]<=dataset.n_users-1 and 0<=x[1]<=dataset.m_items-1 and 0<=x[2]<=dataset.m_items-1)])
    else:
        S = UniformSample_original_python(dataset)
    return S

def UniformSample_original_python(dataset):
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)
