import os
import yaml
from copy import deepcopy

import numpy as np
import scipy.io as sio

import cv2
from PIL import Image
import open3d as o3d

from tqdm import tqdm

from options.scene_render_options import SceneRenderOptions
from utils import utils, mesh_utils, vis_utils

def mat_homog(R_mat, T_mat=np.array([0., 0., 0.])):
    R_mat = np.reshape(R_mat, (3, 3))
    homog_term = np.array([[T_mat[0]], [T_mat[1]], [T_mat[2]]])
    pose = np.concatenate((R_mat, homog_term), axis=1)
    homog_term = np.array([[0., 0., 0., 1.]])
    pose = np.concatenate((pose, homog_term), axis=0)
    return pose

def meshes_place(meshes, meshes_labels_list, poses, is_table=True):
    meshes_list = []
    transed_labels_list = []

    if is_table:
        ref_id = 0
        bias = -6
        # translate table
        temp_vertices = np.asarray(meshes[ref_id].vertices)
        temp_vertices_z = temp_vertices[:, 2]
        min_z = np.min(temp_vertices_z)

        table_mesh = deepcopy(meshes[-1])
        table_mesh = table_mesh.translate((0., 0., min_z + bias))

        # transfrom table same with the ref model
        R_mat = np.array(poses[ref_id]['cam_R_m2c'])
        T_mat = poses[ref_id]['cam_t_m2c']
        pose = mat_homog(R_mat, T_mat)
        table_mesh = table_mesh.transform(pose)

        meshes_list.append(table_mesh)
        transed_labels_list.append(meshes_labels_list[0])

    for i_pose in range(len(poses)):
        obj_id = poses[i_pose]['obj_id']
        i_mesh = meshes_labels_list.index(obj_id) - 1   # get the index of the corresponding mesh stored in meshes

        mesh = deepcopy(meshes[i_mesh])

        R_mat = np.array(poses[i_pose]['cam_R_m2c'])
        T_mat = poses[i_pose]['cam_t_m2c']

        R_mat = np.reshape(R_mat, (3, 3))
        homog_term = np.array([[T_mat[0]], [T_mat[1]], [T_mat[2]]])
        pose = np.concatenate((R_mat, homog_term), axis=1)
        homog_term = np.array([[0., 0., 0., 1.]])
        pose = np.concatenate((pose, homog_term), axis=0)

        mesh = mesh.transform(pose)

        meshes_list.append(mesh)
        transed_labels_list.append(i_mesh + 1)

    return meshes_list, transed_labels_list

def add_table_mesh(meshes_list, trans=np.array([None])):
    table_mesh = o3d.geometry.TriangleMesh()

    '''
    ^y   0___1
    |    |  /|
    |    | / |
    |    |/__|
    |    2   3
    ----->x
    '''
    vertices = np.array([[-720., 720., 0.],
                         [720., 720., 0.],
                         [-720., -720., 0.],
                         [720., -720., 0.]])
    vertex_colors = np.array([[0., 0.447, 0.451],
                              [0., 0.447, 0.451],
                              [0., 0.447, 0.451],
                              [0., 0.447, 0.451]])
    faces = np.array([[0, 2, 1],
                      [1, 2, 3]])

    table_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    table_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    table_mesh.triangles = o3d.utility.Vector3iVector(faces)

    if not trans.any() == None:
        table_mesh = table_mesh.transform(trans)
    
    meshes_list.append(table_mesh)
    return meshes_list

def depth_render(meshes_list, cam_param, output_name='Data/test', depth_scale=1000, is_offscreen=False):
    if is_offscreen:
        vis = o3d.visualization.rendering.OffscreenRenderer(width=640, height=480)
        vis.setup_camera(cam_param.intrinsic, cam_param.extrinsic)
        vis.scene.set_lighting(o3d.visualization.rendering.Open3DScene.LightingProfile.NO_SHADOWS, np.array([0, 0, -1]))
    else:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=640, height=480, visible=False)
        ctr = vis.get_view_control()

    if is_offscreen:
        material = o3d.visualization.rendering.Material()
        for i_mesh in range(len(meshes_list)):
            vis.scene.add_geometry('model_' + str(i_mesh), meshes_list[i_mesh], material)
    else:
        for mesh in meshes_list:
            vis.add_geometry(mesh)

    depth_name = output_name + '_depth.png'
    depth_name_offscreen = output_name + '_depth.tif'
    rgb_name = output_name + '_rgb.png'
    if is_offscreen:
        depth_image = vis.render_to_depth_image()   # Pixels range from 0 (near plane) to 1 (far plane); different from commonly used depth map
        depth_image_np = np.asarray(depth_image) * depth_scale
        depth_image_np = depth_image_np.astype(np.uint16)
        depth_image_pil = Image.fromarray(depth_image_np)
        depth_image_pil.save(depth_name_offscreen)

        rgb_image = vis.render_to_image()
        o3d.io.write_image(rgb_name, rgb_image)
    else:
        ctr.convert_from_pinhole_camera_parameters(cam_param)
        ctr.set_constant_z_far(3000)    # important step
        vis.poll_events()
        vis.capture_depth_image(depth_name, do_render=False, depth_scale=depth_scale)
        vis.capture_screen_image(rgb_name, do_render=False)

def label_render(meshes_list, meshes_labels_list, cam_param, output_name='Data/test', depth_scale=1000, is_offscreen=False):
    if is_offscreen:
        vis = o3d.visualization.rendering.OffscreenRenderer(width=640, height=480)
        vis.setup_camera(cam_param.intrinsic, cam_param.extrinsic)
        vis.scene.set_lighting(o3d.visualization.rendering.Open3DScene.LightingProfile.NO_SHADOWS, np.array([0, 0, -1]))
    else:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=640, height=480, visible=False)
        ctr = vis.get_view_control()

    material = o3d.visualization.rendering.Material()
    for i_mesh in range(len(meshes_labels_list)):
        mesh = meshes_list[i_mesh]
        vertices = np.asarray(mesh.vertices)
        num_vertices = vertices.shape[0]
        mesh_copy = deepcopy(mesh)

        mesh_label = meshes_labels_list[i_mesh]
        label_color = np.ones((num_vertices, 3))
        label_color = label_color * mesh_label * 3 / 255.
        mesh_copy.vertex_colors = o3d.utility.Vector3dVector(label_color)
        if is_offscreen:
            vis.scene.add_geometry('model_' + str(i_mesh), mesh_copy, material)
        else:
            vis.add_geometry(mesh_copy)

    label_name = output_name + '_label.png'
    if is_offscreen:
        rgb_image = vis.render_to_image()
        o3d.io.write_image(label_name, rgb_image)
    else:
        ctr.convert_from_pinhole_camera_parameters(cam_param)
        ctr.set_constant_z_far(3000)
        vis.poll_events()
        vis.capture_screen_image(label_name, do_render=False)

def scene_render(meshes_path, meshes_name, scene_path, output_path='Data/', depth_scale=1000, is_table=True, is_offscreen=False):
    poses_path = scene_path + 'gt.yml'
    intrinsic_path = scene_path + 'info.yml'
    
    with open(poses_path, 'r') as poses_yaml:
        poses_dict = yaml.safe_load(poses_yaml)

    num_frames = len(poses_dict)
    num_meshes = len(poses_dict[0])

    meshes = []
    meshes_labels_list = []

    if is_table:
        meshes_labels_list.append(80)

    for i_mesh in range(num_meshes):
        idx_mesh = str(poses_dict[0][i_mesh]['obj_id']).zfill(2)
        mesh_path = meshes_path + 'obj_' + idx_mesh + meshes_name
        mesh = mesh_utils.load_mesh(mesh_path)
        meshes.append(mesh)
        meshes_labels_list.append(int(idx_mesh))

    if is_table:
        meshes = add_table_mesh(meshes)

    with open(intrinsic_path, 'r') as intrinsic_yaml:
        intrinsics = yaml.safe_load(intrinsic_yaml)
    intrinsic_mat = intrinsics[0]['cam_K']

    # get depth size
    temp_depth = cv2.imread(scene_path + 'depth/0000.png', 2)
    intrinsic_shape = temp_depth.shape

    '''
    K = [[focal * width, 0, width / 2 - 0.5],
         [0, focal * width, height / 2 - 0.5],
         [0, 0, 1]]
    '''
    # intrinsic_param = [intrinsic_shape[1], intrinsic_shape[0], 
    #                    intrinsic_mat[0], intrinsic_mat[4], 
    #                    intrinsic_shape[1] / 2 - 0.5, intrinsic_shape[0] / 2 - 0.5]
    intrinsic_param = [intrinsic_shape[1], intrinsic_shape[0],
                       (intrinsic_mat[0] / (intrinsic_mat[2] * 2)) * intrinsic_shape[1], (intrinsic_mat[4] / (intrinsic_mat[2] * 2)) * intrinsic_shape[1], 
                       intrinsic_shape[1] / 2 - 0.5, intrinsic_shape[0] / 2 - 0.5]

    cam_param = vis_utils.get_camera_parameters(intrinsic_mat=intrinsic_param)

    print('Scene: %s, number of images: %d'%(scene_path, num_frames))

    for i_img in tqdm(range(num_frames)):
        poses = poses_dict[i_img]

        # if is_table:
        #     if not len(poses) == len(meshes) - 1:
        #         continue
        # else:
        #     if not len(poses) == len(meshes):
        #         continue

        meshes_transed, meshes_transed_labels_list = meshes_place(meshes, meshes_labels_list, poses, is_table)

        output_name = output_path + str(i_img)
        # mesh_utils.write_meshes(output_name + '.obj', meshes_transed)
        depth_render(meshes_transed, cam_param, output_name=output_name, depth_scale=depth_scale, is_offscreen=is_offscreen)
        label_render(meshes_transed, meshes_transed_labels_list, cam_param, output_name=output_name, depth_scale=depth_scale, is_offscreen=is_offscreen)

def data_generation(opt):
    scene_list = os.listdir(opt.root_path)

    # # linear processing
    for i_scene in range(len(scene_list)):
        scene_path = opt.root_path + scene_list[i_scene] + '/'
        output_path = opt.output_path + scene_list[i_scene] + '/'
        if not os.path.exists(output_path):
            utils.mkdirs(output_path)
        scene_render(opt.meshes_path, opt.meshes_name, scene_path, output_path, opt.depth_scale, opt.is_table, opt.is_offscreen)

if __name__ == '__main__':
    opt = SceneRenderOptions().parse()

    # test scene render
    scene_render(opt.meshes_path, opt.meshes_name, opt.one_scene_path, output_path=opt.output_path, depth_scale=opt.depth_scale, is_table=opt.is_table)

    # test visualization
    # real_depth = cv2.imread('Data/s0_0_k.png', 2)
    # maxx = np.max(real_depth)
    # real_depth = real_depth / maxx
    # cv2.imshow('00', real_depth)
    # cv2.waitKey(0)

    # real_depth = cv2.imread('Data/depth.png', 2)
    # real_depth = real_depth / maxx
    # cv2.imshow('01', real_depth)
    # cv2.waitKey(0)

    # data_generation(opt)