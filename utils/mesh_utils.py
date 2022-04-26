from copy import deepcopy
from re import A

import numpy as np
import open3d as o3d

def mat_homog(R_mat, T_mat=np.array([0., 0., 0.])):
    R_mat = np.reshape(R_mat, (3, 3))
    homog_term = np.array([[T_mat[0]], [T_mat[1]], [T_mat[2]]])
    pose = np.concatenate((R_mat, homog_term), axis=1)
    homog_term = np.array([[0., 0., 0., 1.]])
    pose = np.concatenate((pose, homog_term), axis=0)
    return pose

def place_meshes_linemod(meshes, meshes_labels_list, poses, is_table=True):
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

def place_meshes_graspnet(meshes, poses, cam_pos=np.array([None])):
    meshes_list = []

    for i_mesh in range(len(meshes)):
        pose = poses[:, :, i_mesh]
        mesh = meshes[i_mesh]

        homog_term = np.array([[0., 0., 0., 1.]])
        pose = np.concatenate((pose, homog_term), axis=0)
        mesh = mesh.transform(pose)

        if not cam_pos.any() == None:
            mesh = mesh.transform(cam_pos)

        meshes_list.append(mesh)

    return meshes_list

def merge_mesh_linemod(meshes):
    num_meshes = len(meshes)

    num_vertices_all = 0
    num_faces_all = 0

    mesh_total = o3d.geometry.TriangleMesh()

    mesh = deepcopy(meshes[0])
    vertices = np.asarray(mesh.vertices)
    vertex_colors = np.asarray(mesh.vertex_colors)
    faces = np.asarray(mesh.triangles)

    vertices_all = vertices
    vertex_colors_all = vertex_colors
    faces_all = faces

    num_vertices_all += vertices.shape[0]
    num_faces_all += faces.shape[0]

    for i_mesh in range(1, num_meshes):
        mesh = deepcopy(meshes[i_mesh])
        vertices = np.asarray(mesh.vertices)
        vertex_colors = np.asarray(mesh.vertex_colors)
        faces = np.asarray(mesh.triangles)

        faces += num_vertices_all

        vertices_all = np.concatenate((vertices_all, vertices), axis=0)
        vertex_colors_all = np.concatenate((vertex_colors_all, vertex_colors), axis=0)
        faces_all = np.concatenate((faces_all, faces), axis=0)

        num_vertices_all += vertices.shape[0]
        num_faces_all += faces.shape[0]

    mesh_total.vertices = o3d.utility.Vector3dVector(vertices_all)
    mesh_total.vertex_colors = o3d.utility.Vector3dVector(vertex_colors_all)
    mesh_total.triangles = o3d.utility.Vector3iVector(faces_all)

    return mesh_total

def merge_mesh_graspnet(meshes, poses):
    num_vertices_all = 0
    num_faces_all = 0

    pose = poses[:, :, 0]
    mesh = meshes[0]

    mesh_total = o3d.geometry.TriangleMesh()

    vertices = np.asarray(mesh.vertices)
    vertex_colors = np.asarray(mesh.vertex_colors)
    faces = np.asarray(mesh.triangles)

    homog_term = np.ones([vertices.shape[0], 1])
    homog_vertices = np.concatenate((vertices, homog_term), axis=1)
    transed_vertices = np.matmul(pose, homog_vertices.T).T  # transpose

    vertices_all = transed_vertices
    vertex_colors_all = vertex_colors
    faces_all = faces

    num_vertices_all += vertices.shape[0]
    num_faces_all += faces.shape[0]

    for i_mesh in range(1, len(meshes)):
        pose = poses[:, :, i_mesh]
        mesh = meshes[i_mesh]
        vertices = np.asarray(mesh.vertices)
        vertex_colors = np.asarray(mesh.vertex_colors)
        faces = np.asarray(mesh.triangles)

        homog_term = np.ones([vertices.shape[0], 1])
        homog_vertices = np.concatenate((vertices, homog_term), axis=1)
        transed_vertices = np.matmul(pose, homog_vertices.T).T

        faces += num_vertices_all

        vertices_all = np.concatenate((vertices_all, transed_vertices), axis=0)
        vertex_colors_all = np.concatenate((vertex_colors_all, vertex_colors), axis=0)
        faces_all = np.concatenate((faces_all, faces), axis=0)

        num_vertices_all += vertices.shape[0]
        num_faces_all += faces.shape[0]

    mesh_total.vertices = o3d.utility.Vector3dVector(vertices_all)
    mesh_total.vertex_colors = o3d.utility.Vector3dVector(vertex_colors_all)
    mesh_total.triangles = o3d.utility.Vector3iVector(faces_all)

    # o3d.visualization.draw_geometries([mesh_total])
    # mesh_utils.write_mesh('Data/output.obj', mesh_total)

    return mesh_total

def load_mesh(mesh_path, is_print=False):
    if is_print:
        print('Loading %s'%mesh_path)

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    return mesh

def write_mesh(mesh_path, mesh, is_print=False):
    if is_print:
        print('Writing %s'%mesh_path)
    o3d.io.write_triangle_mesh(mesh_path, mesh)

def write_meshes(mesh_path, meshes, is_print=False, poses=[]):
    if is_print:
        print('Writing %s'%mesh_path)
    
    if len(poses) == 0:
        mesh = merge_mesh_linemod(meshes)
    else:
        mesh = merge_mesh_graspnet(meshes, poses)

    o3d.io.write_triangle_mesh(mesh_path, mesh)