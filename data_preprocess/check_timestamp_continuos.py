import pandas as pd
from tqdm import tqdm

data = pd.read_csv('../data/BJ_Taxi/chaoyang_traj_201511_input.csv')
prev_time = None
for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    if prev_time is None:
        prev_time = row['time_code']
    else:
        assert prev_time + 1 == row['time_code']
        prev_time = row['time_code']
