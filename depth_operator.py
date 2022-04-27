import os
from copy import deepcopy

import cv2
import numpy as np

import scipy.io as sio
from matplotlib import cm

import open3d as o3d

from options.depth_operator_options import DepthOperatorOptions
from utils import vis_utils

def color_mapping(depth):
    cmap = cm.get_cmap('jet', 256)
    mapped_depth = cmap(depth)
    return mapped_depth

def preprocessing_nonormalize(depth, min_value = 0, max_value = 0):
    processed_depth = depth
    if max_value == 0:
        c_max = 8000
        c_min = 1
    else:
        c_min = min_value
        c_max = max_value

    # c_min = min_value
    processed_depth[processed_depth >= c_max] = c_max
    mask = (processed_depth < 1)

    if not max_value == 0:  # mapping to 0-255
        processed_depth[processed_depth < c_min] = c_min

        processed_depth = processed_depth - c_min
        processed_depth = processed_depth * (255 / (c_max - c_min))

    processed_depth = 255 - processed_depth
    processed_depth = processed_depth.astype(np.int16)
    return processed_depth, mask

def paired_depth_colorization(depth_scan, depth_render):
    # d_max = np.max(depth_scan)
    # missing_mask = (depth_scan < 1)
    # depth_scan[missing_mask] = 10000
    # d_min = np.min(depth_scan)
    # depth_scan[missing_mask] = 0

    d_max = np.max(depth_render) * 1.1
    missing_mask = (depth_render < 1)
    depth_render[missing_mask] = 10000
    d_min = np.min(depth_render)
    depth_render[missing_mask] = 0

    processed_depth_scan, mask_scan = preprocessing_nonormalize(depth_scan, d_min, d_max)
    processed_depth_render, mask_render = preprocessing_nonormalize(depth_render, d_min, d_max)

    mapped_depth_scan = color_mapping(processed_depth_scan)
    mapped_depth_scan[mask_scan] = [0, 0, 0, 1]
    cv2.imwrite('Data/depth_scan_color.png', mapped_depth_scan * 255.)

    mapped_depth_render = color_mapping(processed_depth_render)
    mapped_depth_render[mask_render] = [0, 0, 0, 1]
    cv2.imwrite('Data/depth_render_color.png', mapped_depth_render * 255.)

def get_paried_pointcloud(depth_scan_path, depth_render_path, scene_path, camera='kinect', depth_scale=1000):
    depth_scan = o3d.io.read_image(depth_scan_path)
    depth_render = o3d.io.read_image(depth_render_path)

    cam_pos = np.load(scene_path + 'cam0_wrt_table.npy')
    extrinsic_mat = np.linalg.inv(cam_pos).tolist()
    camara_param = vis_utils.get_camera_parameters(extrinsic_mat, camera=camera)
    
    point_cloud_scan = o3d.geometry.PointCloud.create_from_depth_image(depth_scan, camara_param.intrinsic, camara_param.extrinsic, depth_scale=depth_scale)
    point_cloud_render = o3d.geometry.PointCloud.create_from_depth_image(depth_render, camara_param.intrinsic, camara_param.extrinsic, depth_scale=depth_scale)

    o3d.io.write_point_cloud("Data/point_cloud_scan.xyz", point_cloud_scan)
    o3d.io.write_point_cloud("Data/point_cloud_render.xyz", point_cloud_render)
    o3d.visualization.draw_geometries([point_cloud_scan, point_cloud_render])

def registrate_paried_pointcloud(depth_scan_path, depth_render_path, scene_path, camera='kinect', depth_scale=1000):
    depth_scan = o3d.io.read_image(depth_scan_path)
    depth_render = o3d.io.read_image(depth_render_path)

    cam_pos = np.load(scene_path + 'cam0_wrt_table.npy')
    extrinsic_mat = np.linalg.inv(cam_pos).tolist()
    camara_param = vis_utils.get_camera_parameters(extrinsic_mat, camera=camera)
    
    point_cloud_scan = o3d.geometry.PointCloud.create_from_depth_image(depth_scan, camara_param.intrinsic, camara_param.extrinsic, depth_scale=depth_scale)
    point_cloud_render = o3d.geometry.PointCloud.create_from_depth_image(depth_render, camara_param.intrinsic, camara_param.extrinsic, depth_scale=depth_scale)

    res = o3d.pipelines.registration.registration_icp(point_cloud_render, point_cloud_scan, 5)
    # point_cloud_render_reged = point_cloud_render.transform(res.transformation)

    o3d.io.write_point_cloud("Data/point_cloud_scan.xyz", point_cloud_scan)
    o3d.io.write_point_cloud("Data/point_cloud_render.xyz", point_cloud_render)
    o3d.visualization.draw_geometries([point_cloud_scan, point_cloud_render])

def create_object_pointcloud(masked_depth, camara_param, depth_scale=1000):
    depth = o3d.geometry.Image(masked_depth)
    point_cloud = o3d.geometry.PointCloud.create_from_depth_image(depth, camara_param.intrinsic, camara_param.extrinsic, depth_scale=depth_scale)
    return point_cloud

def label_processing(label_path, is_synthetic=False):
    if not is_synthetic:
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    else:
        label = cv2.imread(label_path)
        label = label[:, :, 0]
        background_mask = (label >= 90) # there are 88 models in total, reduce rasterization error
        label[background_mask] = 0
    return label

def get_object_pointclouds(scene_path, output_path='Data/', camera='kinect', depth_scale=1000, is_synthetic=False):
    depth_paths = os.listdir(scene_path + 'depth/')
    label_paths = os.listdir(scene_path + 'label/')

    cam_pos = np.load(scene_path + 'cam0_wrt_table.npy')
    extrinsic_mat = np.linalg.inv(cam_pos).tolist()
    camara_param = vis_utils.get_camera_parameters(extrinsic_mat, camera=camera)

    for i_img in range(len(depth_paths)):
        depth_path = scene_path + 'depth/' + depth_paths[i_img]
        label_path = scene_path + 'label/' + depth_paths[i_img]
        
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        label_img = label_processing(label_path, is_synthetic)

        pose_label_path = scene_path + 'meta/' + depth_paths[i_img].split('.')[0] + '.mat'
        mat = sio.loadmat(pose_label_path)
        idx_models = mat['cls_indexes'][0]

        for i_label in idx_models:
            depth_mask = (label_img == i_label)
            n_depth_mask = np.logical_not(depth_mask)
            masked_depth = deepcopy(depth_img)
            masked_depth[n_depth_mask] = 0
            point_cloud = create_object_pointcloud(masked_depth, camara_param=camara_param, depth_scale=depth_scale)
            # o3d.visualization.draw_geometries([point_cloud])
            point_cloud_write_path = output_path + 'model_pc/'
            if not os.path.exists(point_cloud_write_path):
                os.makedirs(point_cloud_write_path)
            if is_synthetic:
                point_cloud_write_name = point_cloud_write_path + str(i_img).zfill(4) + '_' + str(i_label).zfill(3) + '_syn.xyz' 
            else:
                point_cloud_write_name = point_cloud_write_path + str(i_img).zfill(4) + '_' + str(i_label).zfill(3) + '.xyz' 
            o3d.io.write_point_cloud(point_cloud_write_name, point_cloud)

if __name__ == '__main__':
    opt = DepthOperatorOptions().parse()

    depth_scan = cv2.imread('Data/graspreal/0000_depth.png', 2)
    depth_render = cv2.imread('Data/graspsyn/0000_depth.png', 2)

    # get_paried_pointcloud('Data/s0_0_k.png', 'Data/depth_kinect.png', opt.one_scene_path, opt.camera, opt.depth_scale)
    paired_depth_colorization(depth_scan, depth_render)
    # get_object_pointclouds(opt.one_scene_path_synthetic, is_synthetic=True)
    
    # registrate_paried_pointcloud('Data/s0_0_r.png', 'Data/depth_realsense.png', 'E:/Datasets/GraspNet/TrainImages/scene_0000/realsense/', 'realsense', opt.depth_scale)