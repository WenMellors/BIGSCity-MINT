# 剔除长度小于 5 的轨迹看下性能

import pandas as pd
from tqdm import tqdm
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

traj_file = '../data/Xian/xianshi_partB_mm_test.csv'
result_file = '../data/Xian/xianshi_partB_mm_test_filtered.csv'

traj = pd.read_csv(traj_file)
long_traj = []
for index, row in tqdm(traj.iterrows(), total=traj.shape[0], desc='count length'):
    rid_list = row['rid_list'].split(',')
    length = len(rid_list)
    long_traj.append(length > 5)

traj = traj[long_traj]
traj.to_csv(result_file, index=False)