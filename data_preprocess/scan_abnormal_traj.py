# 分天统计城市的交通状态，判断某一天是否存在一个异常的流量情况
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

traffic_state_mx = np.load('output_res/traffic_state_mx_201011.npy')
traffic_state_vec = traffic_state_mx.reshape((30, -1))


def visualize(data, labels, noise_color='black', noise_label=-1, title=''):
    '''降维可视化'''
    colors = ['red', 'green', 'blue', 'orange', 'brown', 'gold']
    if len(data[0]) > 2:
        pca = PCA(2)
        data = pca.fit_transform(data)
    unique_labels = np.unique(labels)
    for l, c in zip(unique_labels, colors[:unique_labels.size]):
        points = data[np.where(labels == l)]
        c = noise_color if l == noise_label else c
        for x, y in points:
            plt.scatter(x, y, c=c)
    if title != '':
        plt.title(title)
    plt.show()


# 使用基于余弦距离的 DBSCAN 的算法来做异常检测
dbscan = DBSCAN(eps=0.02, metric='cosine').fit(traffic_state_vec)
labels = dbscan.labels_
print(Counter(dbscan.labels_))
visualize(traffic_state_vec, dbscan.labels_, title='Cosine DBSCAN 201511')
abnormal_day = []
for day in range(30):
    if dbscan.labels_[day] == -1:
        abnormal_day.append(day+1)

print(abnormal_day)
# 2015: 11.14, 11.15, 11.25, 11.27
# 2014: 11.30
# 2012: 11.03, 11.04


