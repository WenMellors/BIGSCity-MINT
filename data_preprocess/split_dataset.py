# 需要按照日期，将数据划分为训练数据与测试数据
import pandas as pd
from tqdm import tqdm
import os
import argparse


def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true'):
        return True
    elif s.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('bool value expected.')


# 引入命令行，更加脚本化
parser = argparse.ArgumentParser()
parser.add_argument('--local', type=str2bool,
                    default=True, help='whether save the trained model')
parser.add_argument('--dataset_name', type=str,
                    default='BJ_Taxi')
parser.add_argument('--data_filename', type=str,
                    default='chaoyang_traj_mm_processed.csv')
parser.add_argument('--train_data_filename', type=str,
                    default='chaoyang_traj_mm_train.csv')
parser.add_argument('--test_data_filename', type=str,
                    default='chaoyang_traj_mm_test.csv')

args = parser.parse_args()
local = args.local

if local:
    data_root = '../data/'
else:
    data_root = '/mnt/data/jwj/'

dataset_name = args.dataset_name
data_filename = args.data_filename
train_data_filename = args.train_data_filename
test_data_filename = args.test_data_filename

raw_data = pd.read_csv(os.path.join(data_root, dataset_name, data_filename))

# 数据集划分比例
if dataset_name == 'BJ_Taxi':
    # 该数据集是2015.11月全月的数据，所以我们 7:3 划分，就是前 21 用于训练和验证，后 9 天用于测试，注意要消除跨天的轨迹
    dividing_date = '2015-11-22'
elif dataset_name == 'Porto_Taxi':
    # 该数据集是2013.07.01到2014.07.01一整年的数据，但是出租车比较少，只有400多辆（大概），还是 7:3 的比例划分
    dividing_date = '2014-03-15'
elif dataset_name == 'Xian':
    # 从 2018.10.31 到 2018.11.30
    dividing_date = '2018-11-22'
else:
    assert dataset_name == 'Chengdu'
    dividing_date = '2018-11-22'

train_data = open(os.path.join(data_root, dataset_name, train_data_filename), 'w')
test_data = open(os.path.join(data_root, dataset_name, test_data_filename), 'w')

train_data.write('traj_id,rid_list,time_list\n')
test_data.write('traj_id,rid_list,time_list\n')
for index, row in tqdm(raw_data.iterrows(), total=raw_data.shape[0]):
    time_list = row['time_list'].split(',')
    start_timestamp = time_list[0]
    if start_timestamp[:10] < dividing_date:
        is_train = True
    else:
        is_train = False
    # 检查是否跨越了分界线
    is_out_range = False
    for timestamp in time_list[1:]:
        if is_train and timestamp[:10] > dividing_date:
            is_out_range = True
            break
    if is_train:
        if not is_out_range:
            # 是训练数据
            train_data.write('{},\"{}\",\"{}\"\n'.format(str(row['traj_id']), row['rid_list'], row['time_list']))
    else:
        test_data.write('{},\"{}\",\"{}\"\n'.format(str(row['traj_id']), row['rid_list'], row['time_list']))
