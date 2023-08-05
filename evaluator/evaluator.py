import json
import os
from tqdm import tqdm
from shapely.geometry import Polygon
# noinspection PyProtectedMember
from shapely.geometry.polygon import orient
import hausdorff
from scipy.spatial.distance import euclidean, cosine, cityblock, chebyshev
from fastdtw import fastdtw
from pyproj import Geod
from scipy.stats import entropy
from geopy import distance
import math
import numpy as np


class Evaluator(object):
    """
    responsible for evaluating the synthetic trajectory
    """

    def __init__(self, data_root, dataset_name, region_name):
        """
        For different dataset, the evaluator should have different settings
        :param data_root:
        :param dataset_name:
        :param region_name:
        """
        self.id2idx = {}  # traj id to the row index of true data
        if dataset_name == 'BJ_Taxi':
            # 海淀区暂时不会实装
            self.road_num = 40306
            self.road_pad = 40306
            with open(os.path.join(data_root, dataset_name, 'region_road_dict.json'), 'r') as f:
                region_road_dict = json.load(f)
            if region_name == 'partA':
                region_road_list = region_road_dict['region_A_road_list']
            else:
                region_road_list = region_road_dict['region_B_road_list']
            self.region_road_idx = {}
            for index, road in enumerate(region_road_list):
                self.region_road_idx[road] = index
            if region_name == 'partA':
                lon_0 = 116.3742330
                lon_1 = 116.5003217
                lat_0 = 39.81009544
                lat_1 = 39.99994671
            else:
                lon_0 = 116.25015100
                lon_1 = 116.38076406
                lat_0 = 39.886962149
                lat_1 = 39.99992161
            # 加载 road2grid 的映射字典
            with open(os.path.join(data_root, dataset_name, '{}_road2grid.json'.format(region_name)), 'r') as f:
                self.road2grid = json.load(f)
        elif dataset_name == 'Porto':
            # Porto Taxi
            self.road_num = 11095
            self.road_pad = 11095
            region_road_list = np.arange(0, self.road_num, 1).tolist()
            self.region_road_idx = {}
            for index, road in enumerate(region_road_list):
                self.region_road_idx[road] = index
            # 定义波尔图的地图大小
            lon_0 = -8.6886557
            lon_1 = -8.5559500
            lat_0 = 41.1405760
            lat_1 = 41.1856278
            # 加载 road2grid 的映射字典
            with open(os.path.join(data_root, dataset_name, 'road2grid.json'), 'r') as f:
                self.road2grid = json.load(f)
        elif dataset_name == 'Xian':
            # Xian
            self.road_num = 17378
            self.road_pad = 17378
            with open(os.path.join(data_root, dataset_name, 'region_road_dict.json'), 'r') as f:
                region_road_dict = json.load(f)
            if region_name == 'partA':
                region_road_list = region_road_dict['region_A_road_list']
            else:
                region_road_list = region_road_dict['region_B_road_list']
            self.region_road_idx = {}
            for index, road in enumerate(region_road_list):
                self.region_road_idx[road] = index
            # 定义西安市的地图大小
            if region_name == 'partA':
                lon_0 = 108.8555107
                lon_1 = 109.0313147
                lat_0 = 34.22585601
                lat_1 = 34.29639324
            else:
                lon_0 = 108.8093989
                lon_1 = 109.0499449
                lat_0 = 34.17026047
                lat_1 = 34.25241200
            # 加载 road2grid 的映射字典
            with open(os.path.join(data_root, dataset_name, '{}_road2grid.json'.format(region_name)), 'r') as f:
                self.road2grid = json.load(f)
        else:
            assert dataset_name == 'Chengdu'
            self.road_num = 28823
            self.road_pad = 28823
            with open(os.path.join(data_root, dataset_name, 'region_road_dict.json'), 'r') as f:
                region_road_dict = json.load(f)
            if region_name == 'partA':
                region_road_list = region_road_dict['region_A_road_list']
            else:
                region_road_list = region_road_dict['region_B_road_list']
            self.region_road_idx = {}
            for index, road in enumerate(region_road_list):
                self.region_road_idx[road] = index
            # 定义成都市的地图大小
            if region_name == 'partA':
                lon_0 = 103.9024499
                lon_1 = 104.0914319
                lat_0 = 30.52970855
                lat_1 = 30.71864067
            else:
                lon_0 = 104.0586068
                lon_1 = 104.2261992
                lat_0 = 30.54882405
                lat_1 = 30.74048714
            # 加载 road2grid 的映射字典
            with open(os.path.join(data_root, dataset_name, '{}_road2grid.json'.format(region_name)), 'r') as f:
                self.road2grid = json.load(f)
        self.img_unit = 0.005  # 这样画出来的大小，大概是 0.42 km * 0.55 km 的格子
        self.img_width = math.ceil((lon_1 - lon_0) / self.img_unit) + 1  # 图像的宽度
        self.img_height = math.ceil((lat_1 - lat_0) / self.img_unit) + 1  # 映射出的图像的高度
        self.total_grid = self.img_height * self.img_width
        with open(os.path.join(data_root, dataset_name, 'rid_gps.json'), 'r') as f:
            self.road_gps = json.load(f)
        # 设置与 bins 化相关的参数
        self.max_distance = 100  # 这里设置一个出行上限阈值
        self.max_radius = 31.6764 * 31.6764

    def evaluate_data(self, true_data, test_data, test_mode=0):
        """
        evaluate the similarity between true_data and test_data
        :param true_data:
        :param test_data:
        :return: result
        """
        # we first count the macro metrics distribution of true data
        true_travel_distance_total = []
        true_travel_radius_total = []
        true_location_cnt = np.zeros(len(self.region_road_idx), dtype=np.int)
        true_grid_od_cnt = np.zeros((self.total_grid, self.total_grid), dtype=np.int)
        real_max_distance = 0
        real_max_radius = 0  # 这个值是真实数据中的上限
        for index, row in tqdm(true_data.iterrows(), total=true_data.shape[0], desc='count true trajectory'):
            if test_mode == 0:
                traj_id = row['traj_id']
                self.id2idx[traj_id] = index
            else:
                # 直接按照索引拿
                self.id2idx[index] = index
            rid_list = [int(x) for x in row['rid_list'].split(',')]
            travel_distance = 0
            pre_gps = None
            rid_lat = []
            rid_lon = []
            # 计算轨迹的 OD 流
            if str(rid_list[0]) not in self.road2grid:
                # 这是条废物轨迹，不做统计
                # will happen?
                continue
            start_rid_grid = self.road2grid[str(rid_list[0])]
            des_rid_grid = self.road2grid[str(rid_list[-1])]
            start_rid_grid_index = start_rid_grid[0] * self.img_height + start_rid_grid[1]
            des_rid_grid_index = des_rid_grid[0] * self.img_height + des_rid_grid[1]
            true_grid_od_cnt[start_rid_grid_index][des_rid_grid_index] += 1
            for rid_index, rid in enumerate(rid_list):
                # 考虑到某些方法生成的轨迹路段并不邻接，所以这里使用轨迹的 GPS 来计算距离
                gps = self.road_gps[str(rid)]
                if pre_gps is None:
                    pre_gps = gps
                else:
                    travel_distance += distance.distance((gps[1], gps[0]), (pre_gps[1], pre_gps[0])).kilometers
                    pre_gps = gps
                rid_lat.append(gps[1])
                rid_lon.append(gps[0])
                true_location_cnt[self.region_road_idx[rid]] += 1  # 这里需要套用 region_road_idx 重新索引一下
            real_max_distance = max(real_max_distance, travel_distance)
            true_travel_distance_total.append(travel_distance)
            # 计算 radius
            travel_radius = get_geogradius(rid_lat=rid_lat, rid_lon=rid_lon)
            real_max_radius = max(real_max_radius, travel_radius)
            true_travel_radius_total.append(travel_radius)
        # 遍历生成数据，统计其宏观特性并计算微观相似度
        test_travel_distance_total = []
        test_travel_radius_total = []
        test_location_cnt = np.zeros(len(self.region_road_idx), dtype=np.int)
        test_grid_od_cnt = np.zeros((self.total_grid, self.total_grid), dtype=np.int)
        # 微观相似度
        total_edit_distance = 0
        total_edit_distance_on_real = 0
        total_hausdorff = 0
        total_dtw = 0
        for index, row in tqdm(test_data.iterrows(), total=test_data.shape[0], desc='count test trajectory'):
            if index >= true_data.shape[0]:
                break
            if test_mode == 0:
                traj_id = row['traj_id']
                # 索引对应的真实轨迹数据
                true_data_idx = self.id2idx[traj_id]
                true_row = true_data.iloc[true_data_idx]
            else:
                true_data_idx = self.id2idx[index]
                true_row = true_data.iloc[true_data_idx]
            true_rid_list = [int(x) for x in true_row['rid_list'].split(',')]
            true_gps_list = []
            for rid in true_rid_list:
                now_gps = self.road_gps[str(rid)]
                true_gps_list.append([now_gps[1], now_gps[0]])
            rid_list = [int(float(x)) for x in row['rid_list'].split(',')]
            # 删除补齐值
            rid_list = np.array(rid_list)
            rid_list = rid_list[rid_list != self.road_pad].tolist()
            # 需要检查路段是否生成出了 region_road_idx
            clean_rid_list = []
            for rid in rid_list:
                if rid in self.region_road_idx:
                    clean_rid_list.append(rid)
                else:
                    break
            rid_list = clean_rid_list
            if len(rid_list) == 0:
                continue
            # 计算编辑距离
            total_edit_distance += edit_distance(rid_list, true_rid_list)
            travel_distance = 0
            pre_gps = None
            rid_lat = []
            rid_lon = []
            generate_gps_list = []
            # 计算轨迹的 OD 流
            if str(rid_list[0]) not in self.road2grid:
                # 这是条废物轨迹，不做统计
                # will happen?
                continue
            start_rid_grid = self.road2grid[str(rid_list[0])]
            des_rid_grid = self.road2grid[str(rid_list[-1])]
            start_rid_grid_index = start_rid_grid[0] * self.img_height + start_rid_grid[1]
            des_rid_grid_index = des_rid_grid[0] * self.img_height + des_rid_grid[1]
            test_grid_od_cnt[start_rid_grid_index][des_rid_grid_index] += 1
            # 遍历轨迹计算出行距离与 radius
            for rid_index, rid in enumerate(rid_list):
                # 考虑到某些方法生成的轨迹路段并不邻接，所以这里使用轨迹的 GPS 来计算距离
                gps = self.road_gps[str(rid)]
                generate_gps_list.append([gps[1], gps[0]])
                if pre_gps is None:
                    pre_gps = gps
                else:
                    travel_distance += distance.distance((gps[1], gps[0]), (pre_gps[1], pre_gps[0])).kilometers
                    pre_gps = gps
                rid_lat.append(gps[1])
                rid_lon.append(gps[0])
                test_location_cnt[self.region_road_idx[rid]] += 1  # 这里需要套用 region_road_idx 重新索引一下
            real_max_distance = max(real_max_distance, travel_distance)
            test_travel_distance_total.append(travel_distance)
            # 计算 radius
            travel_radius = get_geogradius(rid_lat=rid_lat, rid_lon=rid_lon)
            real_max_radius = max(real_max_radius, travel_radius)
            test_travel_radius_total.append(travel_radius)
            # 计算剩下三个距离
            true_gps_list = np.array(true_gps_list)
            generate_gps_list = np.array(generate_gps_list)
            total_hausdorff += hausdorff_metric(true_gps_list, generate_gps_list)
            total_dtw += dtw_metric(true_gps_list, generate_gps_list)
            total_edit_distance_on_real += s_edr(true_gps_list, generate_gps_list)
        # 计算最终的微观相似度
        total_traj = test_data.shape[0]
        avg_edt = total_edit_distance / total_traj
        avg_edr = total_edit_distance_on_real / total_traj
        avg_hausdorff = total_hausdorff / total_traj
        avg_dtw = total_dtw / total_traj
        # 计算最终的宏观相似度
        # 构建 bins
        # 手动构建 distance_bins, radius_bins
        real_max_distance += 1e-6
        real_max_radius += 1e-6
        travel_distance_bins = np.arange(0, real_max_distance, float(real_max_distance) / 1000).tolist()
        # 将从 real max distance 到 max_distance 设置为一个 bin
        travel_distance_bins.append(real_max_distance + 1)
        if self.max_distance > real_max_distance:
            travel_distance_bins.append(self.max_distance)
        else:
            travel_distance_bins.append(real_max_distance + 2)
        print('real_max_distance', real_max_distance)
        print('self.max_distance', self.max_distance)
        # assert self.max_distance > real_max_distance
        travel_distance_bins = np.array(travel_distance_bins)
        travel_radius_bins = np.arange(0, real_max_radius, float(real_max_radius) / 100).tolist()
        travel_radius_bins.append(real_max_radius + 1)
        if self.max_radius > real_max_radius:
            travel_radius_bins.append(self.max_radius)
        else:
            travel_radius_bins.append(real_max_radius + 2)
        print('real_max_radius', real_max_radius)
        print('self.max_radius', self.max_radius)
        # assert real_max_radius < self.max_radius
        travel_radius_bins = np.array(travel_radius_bins)
        # 计算 travel_distance 与 travel_radius 的分布
        true_travel_distance_distribution, _ = np.histogram(true_travel_distance_total, travel_distance_bins)
        true_travel_radius_distribution, _ = np.histogram(true_travel_radius_total, travel_radius_bins)
        test_travel_distance_distribution, _ = np.histogram(test_travel_distance_total, travel_distance_bins)
        test_travel_radius_distribution, _ = np.histogram(test_travel_radius_total, travel_radius_bins)
        travel_distance_js = js_divergence(true_travel_distance_distribution, test_travel_distance_distribution)
        travel_radius_js = js_divergence(true_travel_radius_distribution, test_travel_radius_distribution)
        # 计算 Grid OD 相似度
        grid_od_js = js_divergence(true_grid_od_cnt.flatten(), test_grid_od_cnt.flatten())
        # 计算 Location 相似度
        location_js = js_divergence(true_location_cnt, test_location_cnt)
        return {'avg_edt': avg_edt, 'avg_hausdorff': avg_hausdorff, 'avg_dtw': avg_dtw,
                'travel_distance_js': travel_distance_js, 'travel_radius_js': travel_radius_js,
                'grid_od_js': grid_od_js, 'location_js': location_js, 'avg_edr': avg_edr}


def cal_polygon_area(polygon, mode=1):
    """
    计算经纬度多边形的覆盖面积（平方米不会算，先用平方度来做）

    Args:
        polygon (list): 多边形顶点经纬度数组
        mode (int): 1: 平方度， 2：平方千米

    Returns:
        area (float)
    """
    if mode == 1:
        if len(polygon) < 3:
            return 0
        area = Polygon(polygon)
        return area.area
    else:
        if len(polygon) < 3:
            return 0
        geod = Geod(ellps="WGS84")
        area, _ = geod.geometry_area_perimeter(orient(Polygon(polygon)))  # 单位平方米
        return area / 1000000


def arr_to_distribution(arr, arr_min, arr_max, bins=10000):
    """
    convert an array to a probability distribution
    :param arr: np.array, input array
    :param arr_min: float, minimum of converted value
    :param arr_max: float, maximum of converted value
    :param bins: int, number of bins between min and max
    :return: np.array, output distribution array
    """
    distribution, base = np.histogram(
        arr, np.arange(
            arr_min, arr_max, float(
                arr_max - arr_min) / bins))
    return distribution


def get_geogradius(rid_lat, rid_lon):
    """
    get the std of the distances of all points away from center as `gyration radius`
    Args:
        rid_lat:
        rid_lon:
    Returns:

    """
    if len(rid_lat) == 0:
        return 0
    lng1, lat1 = np.mean(rid_lon), np.mean(rid_lat)
    rad = []
    for i in range(len(rid_lat)):
        lng2 = rid_lon[i]
        lat2 = rid_lat[i]
        dis = distance.distance((lat1, lng1), (lat2, lng2)).kilometers
        rad.append(dis)
    rad = np.mean(rad)
    return rad

def js_divergence(p, q):
    """JS散度

    Args:
        p(np.array):
        q(np.array):

    Returns:

    """
    m = (p + q) / 2
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)


# def grid_distribution_map(grid_cnt, name1='grid_distribution'):
#     """
#
#     Args:
#         grid_cnt (np.array): shape (img_width, img_height) 网格访问频次
#         name1 (string): 分布图的名字
#
#     Returns:
#
#     """
#     geojson = dict()
#     geojson['type'] = 'FeatureCollection'
#     obj_list = []
#     # 计算各网格的坐标
#     grid_coordinates = []
#     grid_xy = []
#     for x in range(grid_cnt.shape[0]):
#         for y in range(grid_cnt.shape[1]):
#             x_0 = lon_0 + img_unit * x
#             y_0 = lat_0 + img_unit * y
#             x_1 = x_0 + img_unit
#             y_1 = y_0 + img_unit
#             coordinates = [[x_0, y_0], [x_1, y_0], [x_1, y_1], [x_0, y_1], [x_0, y_0]]
#             grid_coordinates.append(coordinates)
#             grid_xy.append((x, y))
#     for (i, grid) in enumerate(grid_coordinates):
#         obj = dict()
#         obj['type'] = 'Feature'
#         obj['properties'] = dict()
#         obj['properties']['cnt'] = int(grid_cnt[grid_xy[i][0], grid_xy[i][1]])
#         obj['geometry'] = dict()
#         obj['geometry']['type'] = 'Polygon'
#         obj['geometry']['coordinates'] = [grid]
#         obj_list.append(obj)
#     geojson['features'] = obj_list
#     json.dump(geojson, open('data/temp_geojson.json', 'w'))
#
#     gpd_geojson = gpd.read_file('data/temp_geojson.json')
#
#     gpd_geojson.plot('cnt', legend=True, cmap=plt.cm.Reds)
#     plt.title(name1)
#     plt.savefig('./save/test_result/{}_geojson.png'.format(name1))


def edit_distance(trace1, trace2):
    """
    the edit distance between two trajectory
    Args:
        trace1:
        trace2:
    Returns:
        edit_distance
    """
    matrix = [[i + j for j in range(len(trace2) + 1)] for i in range(len(trace1) + 1)]
    for i in range(1, len(trace1) + 1):
        for j in range(1, len(trace2) + 1):
            if trace1[i - 1] == trace2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
    return matrix[len(trace1)][len(trace2)]


def hausdorff_metric(truth, pred, distance='haversine'):
    """豪斯多夫距离
    ref: https://github.com/mavillan/py-hausdorff

    Args:
        truth: 经纬度点，(trace_len, 2)
        pred: 经纬度点，(trace_len, 2)
        distance: dist计算方法，包括haversine，manhattan，euclidean，chebyshev，cosine

    Returns:

    """
    return hausdorff.hausdorff_distance(truth, pred, distance=distance)


def haversine(array_x, array_y):
    R = 6378.0
    radians = np.pi / 180.0
    lat_x = radians * array_x[0]
    lon_x = radians * array_x[1]
    lat_y = radians * array_y[0]
    lon_y = radians * array_y[1]
    dlon = lon_y - lon_x
    dlat = lat_y - lat_x
    a = (pow(math.sin(dlat/2.0), 2.0) + math.cos(lat_x) * math.cos(lat_y) * pow(math.sin(dlon/2.0), 2.0))
    return R * 2 * math.asin(math.sqrt(a))


def dtw_metric(truth, pred, distance='haversine'):
    """动态时间规整算法
    ref: https://github.com/slaypni/fastdtw

    Args:
        truth: 经纬度点，(trace_len, 2)
        pred: 经纬度点，(trace_len, 2)
        distance: dist计算方法，包括haversine，manhattan，euclidean，chebyshev，cosine

    Returns:

    """
    if distance == 'haversine':
        distance, path = fastdtw(truth, pred, dist=haversine)
    elif distance == 'manhattan':
        distance, path = fastdtw(truth, pred, dist=cityblock)
    elif distance == 'euclidean':
        distance, path = fastdtw(truth, pred, dist=euclidean)
    elif distance == 'chebyshev':
        distance, path = fastdtw(truth, pred, dist=chebyshev)
    elif distance == 'cosine':
        distance, path = fastdtw(truth, pred, dist=cosine)
    else:
        distance, path = fastdtw(truth, pred, dist=euclidean)
    return distance


rad = math.pi / 180.0
R = 6378137.0


def great_circle_distance(lat1, lon1, lat2, lon2):
    """
    Usage
    -----
    Compute the great circle distance, in meter, between (lat1,lon1) and (lat2,lon2)

    Parameters
    ----------
    lat1: float, latitude of the first point
    lon1: float, longitude of the first point
    lat2: float, latitude of the se*cond point
    lon2: float, longitude of the second point

    Returns
    --------
    d: float, Great circle distance between (lon1,lat1) and (lon2,lat2)
    """

    dlat = rad * (lat2 - lat1)
    dlon = rad * (lon2 - lon1)
    a = (math.sin(dlat / 2.0) * math.sin(dlat / 2.0) +
         math.cos(rad * lat1) * math.cos(rad * lat2) *
         math.sin(dlon / 2.0) * math.sin(dlon / 2.0))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d


def s_edr(t0, t1, eps=100.0):
    """
    Usage
    -----
    The Edit Distance on Real sequence between trajectory t0 and t1.

    Parameters
    ----------
    t0 : len(t0)x2 numpy_array
    t1 : len(t1)x2 numpy_array
    eps : float, eps distance in meter

    Returns
    -------
    edr : float
           The Longuest-Common-Subsequence distance between trajectory t0 and t1
    """
    n0 = len(t0)
    n1 = len(t1)
    # An (m+1) times (n+1) matrix
    # C = [[0] * (n1 + 1) for _ in range(n0 + 1)]
    C = np.full((n0 + 1, n1 + 1), np.inf)
    C[:, 0] = np.arange(n0 + 1)
    C[0, :] = np.arange(n1 + 1)
    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            if great_circle_distance(t0[i - 1][0], t0[i - 1][1], t1[j - 1][0], t1[j - 1][1]) < eps:
                subcost = 0
            else:
                subcost = 1
            C[i][j] = min(C[i][j - 1] + 1, C[i - 1][j] + 1, C[i - 1][j - 1] + subcost)
    edr = float(C[n0][n1]) / max([n0, n1])
    return edr


def rid_cnt2heat_level(rid_cnt):
    cnt_max = np.max(rid_cnt)
    level_num = 100
    bin_size = cnt_max // level_num
    rid_heat_level = rid_cnt // bin_size
    return rid_heat_level

