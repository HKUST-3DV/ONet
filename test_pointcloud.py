import os
import sys
import plyfile
import numpy as np
from im2mesh.utils.io import export_pointcloud

def main():
    pcl_filepath = '/media/ziqianbai/BACKPACK_DATA1/DeepPanoContext/iGibson/aug_igibson_obj/chair/1b6c268811e1724ead75d368738e0b47/for_occnet/pointcloud.npz'
    
    pointcloud_dict = np.load(pcl_filepath)
    points = pointcloud_dict['points'].astype(np.float32)
    normals = pointcloud_dict['normals'].astype(np.float32)
    print('points size {}'.format(points.shape))

    new_pcl_filepath = '/media/ziqianbai/BACKPACK_DATA1/DeepPanoContext/iGibson/aug_igibson_obj/chair/1b6c268811e1724ead75d368738e0b47/for_occnet/pointcloud.ply'
    export_pointcloud(points, new_pcl_filepath)



if __name__ == '__main__':
    main()