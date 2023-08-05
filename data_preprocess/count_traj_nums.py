import pandas as pd

filename = ['/mnt/data/jwj/Xian/xianshi_partA_mm_train.csv',
            '/mnt/data/jwj/Xian/xianshi_partA_mm_test.csv',
            '/mnt/data/jwj/Xian/xianshi_partB_mm_train.csv',
            '/mnt/data/jwj/Xian/xianshi_partB_mm_test.csv',
            '/mnt/data/jwj/Chengdu/chengdushi_partA_mm_train.csv',
            '/mnt/data/jwj/Chengdu/chengdushi_partA_mm_test.csv',
            '/mnt/data/jwj/Chengdu/chengdushi_partB_mm_train.csv',
            '/mnt/data/jwj/Chengdu/chengdushi_partB_mm_test.csv']

for file in filename:
    df = pd.read_csv(file)
    print(file, df.shape[0])

