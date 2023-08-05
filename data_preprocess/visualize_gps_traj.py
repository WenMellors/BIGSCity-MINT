import json
import numpy as np
from tqdm import tqdm

with open('../data/Xian/error_id_0.json', 'r') as f:
    error_id_1 = json.load(f)

mm_id = np.random.choice(error_id_1['outside'], size=1).item()
print(mm_id)
geojson = dict()
geojson['type'] = 'FeatureCollection'
obj_list = []

gps_rid_list = []
with open('../data/Xian/xianshi_partA_gps_traj.csv', 'r') as f:
    f.readline()
    start = False
    for line in tqdm(f):
        items = line.split(';')
        id = int(items[0])
        if id == mm_id:
            gps_rid_list.append([float(items[1]), float(items[2])])
            start = True
        elif start:
            break

obj = {'type': 'Feature', 'geometry': {}, 'properties': {'traj_id': mm_id}}
obj['geometry']['type'] = 'LineString'
obj['geometry']['coordinates'] = gps_rid_list
obj_list.append(obj)

geojson['features'] = obj_list
with open('../data/Xian/gps_traj_{}.json'.format(mm_id), 'w') as f:
    json.dump(geojson, f)
