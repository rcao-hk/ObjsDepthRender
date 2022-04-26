from copy import deepcopy

import numpy as np
import open3d as o3d

def merge_mesh(meshes):
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

def load_mesh(mesh_path, is_print=False):
    if is_print:
        print('Loading %s'%mesh_path)

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    return mesh

def write_mesh(mesh_path, mesh, is_print=False):
    if is_print:
        print('Writing %s'%mesh_path)
    o3d.io.write_triangle_mesh(mesh_path, mesh)

def write_meshes(mesh_path, meshes, is_print=False):
    if is_print:
        print('Writing %s'%mesh_path)

    mesh = merge_mesh(meshes)
    o3d.io.write_triangle_mesh(mesh_path, mesh)