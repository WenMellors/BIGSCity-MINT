# 计算一个轨迹数据集的统计特性
import pandas as pd
from evaluator.evaluator import Evaluator
from util.parser import str2bool
import os
import argparse

evaluate_true = False
parser = argparse.ArgumentParser()
parser.add_argument('--local', type=str2bool, default=False)
parser.add_argument('--dataset_name', type=str, default='BJ_Taxi')
parser.add_argument('--region_name', type=str, default='partA')
parser.add_argument('--debug', type=str2bool, default=False)
parser.add_argument('--test_model_version', type=str, default='agent_imitate_learning')
parser.add_argument('--test_mode', type=int, default=0)

args = parser.parse_args()
local = args.local
dataset_name = args.dataset_name
debug = args.debug
region_name = args.region_name
test_model_version = args.test_model_version
test_mode = args.test_mode

if local:
    data_root = './data/'
else:
    data_root = '/mnt/data/jwj/'
# 创建保存评测结果的文件夹
test_res_save_folder = './save/{}/test_result/'.format(dataset_name)
if not os.path.exists(test_res_save_folder):
    os.makedirs(test_res_save_folder)

if dataset_name == 'BJ_Taxi':
    # 暂时不实装海淀区
    if region_name == 'partA':
        true_data = pd.read_csv(os.path.join(data_root, dataset_name, 'beijing_partA_mm_test.csv'))
    else:
        true_data = pd.read_csv(os.path.join(data_root, dataset_name, 'beijing_partB_mm_test.csv'))
    test_data = pd.read_csv(os.path.join(data_root, dataset_name, '{}_{}_generate.csv'.format(test_model_version,
                                                                                              region_name)))
    test_result_filename = os.path.join(test_res_save_folder, '{}_{}_result.pt'.format(test_model_version,
                                                                                       region_name))
elif dataset_name == 'Porto_Taxi':
    # Porto Taxi
    true_data = pd.read_csv(os.path.join(data_root, dataset_name, 'porto_mm_test.csv'))
    test_data = pd.read_csv(os.path.join(data_root, dataset_name, '{}_generate.csv'.format(test_model_version)))
    test_result_filename = os.path.join(test_res_save_folder, '{}_result.pt'.format(test_model_version))
elif dataset_name == 'Xian':
    # Xian
    if region_name == 'partA':
        true_data = pd.read_csv(os.path.join(data_root, dataset_name, 'xianshi_partA_mm_test.csv'))
    else:
        true_data = pd.read_csv(os.path.join(data_root, dataset_name, 'xianshi_partB_mm_test.csv'))
    test_data = pd.read_csv(os.path.join(data_root, dataset_name, '{}_{}_generate.csv'.format(test_model_version,
                                                                                              region_name)))
    test_result_filename = os.path.join(test_res_save_folder, '{}_{}_result.pt'.format(test_model_version,
                                                                                       region_name))
else:
    assert dataset_name == 'Chengdu'
    if region_name == 'partA':
        true_data = pd.read_csv(os.path.join(data_root, dataset_name, 'chengdushi_partA_mm_test.csv'))
    else:
        true_data = pd.read_csv(os.path.join(data_root, dataset_name, 'chengdushi_partB_mm_test.csv'))
    test_data = pd.read_csv(os.path.join(data_root, dataset_name, '{}_{}_generate.csv'.format(test_model_version,
                                                                                              region_name)))
    test_result_filename = os.path.join(test_res_save_folder, '{}_{}_result.pt'.format(test_model_version,
                                                                                       region_name))

# 加载评估器
evaluator = Evaluator(data_root=data_root, dataset_name=dataset_name, region_name=region_name)
# Evaluate
result = evaluator.evaluate_data(true_data=true_data, test_data=test_data, test_mode=test_mode)
print(result)
