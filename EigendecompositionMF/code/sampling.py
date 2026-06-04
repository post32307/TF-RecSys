import world
import numpy as np
from dataloader import BasicDataset
from time import time

try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(world.seed)
    sample_ext = True
except:
    world.cprint("Cpp extension not loaded")
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
