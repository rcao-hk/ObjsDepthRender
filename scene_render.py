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
from utils import mesh_utils, vis_utils

def add_table_mesh(meshes_list, trans=np.array([None]), scale=1.):
    table_mesh = o3d.geometry.TriangleMesh()

    '''
    ^y   0___1
    |    |  /|
    |    | / |
    |    |/__|
    |    2   3
    ----->x
    '''
    vertices = np.array([[-scale, scale, 0.],
                         [scale, scale, 0.],
                         [-scale, -scale, 0.],
                         [scale, -scale, 0.]])
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

def depth_render(meshes_list, cam_param, output_name='Data/test',
                depth_scale=1000, width=1280, height=720, is_offscreen=False):
    if is_offscreen:
        vis = o3d.visualization.rendering.OffscreenRenderer(width=width, height=height)
        vis.setup_camera(cam_param.intrinsic, cam_param.extrinsic)
        vis.scene.set_lighting(o3d.visualization.rendering.Open3DScene.LightingProfile.NO_SHADOWS, np.array([0, 0, -1]))
    else:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height, visible=False)
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
        depth_image = vis.render_to_depth_image()   # Pixels range from 0 (near plane) to 1 (far plane); 
                                                    # different from commonly used depth map
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

def label_render(meshes_list, meshes_labels_list, cam_param, output_name='Data/test',
                depth_scale=1000, width=1280, height=720, is_offscreen=False):
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

def scene_render_linemod(meshes_path, meshes_name, scene_path, output_path='Data/', 
                        depth_scale=1000, output_width=640, output_height=480,
                        is_table=True, is_offscreen=False):
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
        meshes = add_table_mesh(meshes, scale=720.)

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
                       (intrinsic_mat[0] / (intrinsic_mat[2] * 2)) * intrinsic_shape[1],
                       (intrinsic_mat[4] / (intrinsic_mat[2] * 2)) * intrinsic_shape[1], 
                       intrinsic_shape[1] / 2 - 0.5, intrinsic_shape[0] / 2 - 0.5]

    cam_param = vis_utils.get_camera_parameters(intrinsic_mat=intrinsic_param)

    print('Scene: %s, number of images: %d'%(scene_path, num_frames))

    for i_img in tqdm(range(num_frames)):
        poses = poses_dict[i_img]

        meshes_transed, meshes_transed_labels_list = mesh_utils.place_meshes_linemod(meshes, meshes_labels_list, poses, is_table)

        output_name = output_path + str(i_img)
        # mesh_utils.write_meshes(output_name + '.obj', meshes_transed)
        depth_render(meshes_transed, cam_param, output_name=output_name,
                    depth_scale=depth_scale, width=output_width, height=output_height, is_offscreen=is_offscreen)
        label_render(meshes_transed, meshes_transed_labels_list, cam_param, output_name=output_name,
                    depth_scale=depth_scale, width=output_width, height=output_height, is_offscreen=is_offscreen)

def scene_render_graspnet(meshes_path, meshes_name, scene_path, output_path='Data/', camera='kinect',
                        depth_scale=1000, output_width=1280, output_height=720,
                        is_table=True, is_offscreen=False):
    poses_paths = os.listdir(scene_path + 'meta/')
    cam_pos = np.load(scene_path + 'cam0_wrt_table.npy')
    extrinsic_mat = np.linalg.inv(cam_pos).tolist()
    cam_trans_poses = np.load(scene_path + 'camera_poses.npy')

    if is_table:
        cam_param = vis_utils.get_type_camera_parameters(extrinsic_mat, camera=camera)
    else:
        cam_param = vis_utils.get_type_camera_parameters(camera=camera)

    print('Scene: %s, number of images: %d'%(scene_path, len(poses_paths)))

    meshes_transed = []
    meshes_labels_list = []
    # for i_img in tqdm(range(len(poses_paths))):
    for i_img in range(len(poses_paths)):
        meshes = []
        pose_idx = poses_paths[i_img].split('.')[0]
        pose_path = scene_path + 'meta/' + poses_paths[i_img]
        mat = sio.loadmat(pose_path)
        poses = mat['poses']

        num_meshes = poses.shape[2]
        
        idx_meshes = mat['cls_indexes']
        for i_mesh in range(num_meshes):
            idx_mesh = str(idx_meshes[0][i_mesh] - 1).zfill(3)
            mesh_path = meshes_path + idx_mesh + '/' + meshes_name
            mesh = mesh_utils.load_mesh(mesh_path)
            meshes.append(mesh)
            if i_img == 0:
                meshes_labels_list.append(int(idx_mesh) + 1)
        
        if is_table:
            meshes_transed = mesh_utils.place_meshes_graspnet(meshes, poses, cam_pos)
            meshes_transed = add_table_mesh(meshes_transed)
            break
        else:
            meshes_transed = mesh_utils.place_meshes_graspnet(meshes, poses)

        # o3d.visualization.draw_geometries(meshes_list)
        output_name = output_path + pose_idx
        depth_render(meshes_transed, cam_param, output_name=output_name,
                    depth_scale=depth_scale, width=output_width, height=output_height, is_offscreen=is_offscreen)
        label_render(meshes_transed, meshes_labels_list, cam_param, output_name=output_name,
                    depth_scale=depth_scale, width=output_width, height=output_height, is_offscreen=is_offscreen)

        print('    Image: %d, number of meshes: %d'%(i_img, num_meshes))

def data_generation(opt):
    scene_list = os.listdir(opt.root_path)

    # linear processing
    for i_scene in range(len(scene_list)):
        scene_path = opt.root_path + scene_list[i_scene] + '/'
        output_path = opt.output_path + scene_list[i_scene] + '/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if opt.dataset == 'linemod':
            scene_render_linemod(opt.meshes_path, opt.meshes_name, scene_path, output_path, 
                                opt.depth_scale, opt.output_width, opt.output_height,
                                opt.is_table, opt.is_offscreen)
        elif opt.dataset == 'graspnet':
            scene_render_graspnet(opt.meshes_path, opt.meshes_name, scene_path, output_path,
                                opt.camera, opt.depth_scale, opt.output_width, opt.output_height,
                                opt.is_table, opt.is_offscreen)
if __name__ == '__main__':
    opt = SceneRenderOptions().parse()

    # test scene render
    # scene_render_linemod(opt.meshes_path, opt.meshes_name, opt.one_scene_path, output_path=opt.output_path,
    #                     depth_scale=opt.depth_scale, is_table=opt.is_table)

    data_generation(opt)