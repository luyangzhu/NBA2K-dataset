import os
import os.path as osp
from glob import glob
import re
import math
import numpy as np
import cv2

def writeObj(vertices, src_path, dst_path):
    src_obj = open(src_path,'r')
    dst_obj = open(dst_path, 'w')
    src_obj_content = src_obj.readlines()
    
    # write header
    dst_obj.write(src_obj_content[0])
    dst_obj.write(src_obj_content[1])
    dst_obj.write(src_obj_content[2])
    dst_obj.write(src_obj_content[3])
    dst_obj.write(src_obj_content[4])
    dst_obj.write(src_obj_content[5])

    
    # Write vertices
    dst_obj.write("\n# Vertex positions\n")
    for vertex in vertices:
        dst_obj.write("v %.5f %.5f %.5f\n" %(vertex[0], vertex[1], vertex[2]))

    # Write texcoords and faces
    for line in src_obj_content:
        if line.split(' ')[0] == 'vt':
            dst_obj.write(line)
        if line.split(' ')[0] == 'f':
            dst_obj.write(line)
        if line.split(' ')[0] == 'g':
            dst_obj.write(line)
        if line.split(' ')[0] == 'usemtl':
            dst_obj.write(line)

def readObjV2(file_path):
    obj_file = open(file_path,'r')
    contents = obj_file.readlines()
    v,vt,v_idx,vt_idx = [],[],[],[]
    for line in contents:
        parts = line.split(' ')
        if parts[0] == 'v':
            vertex = [float(i) for i in parts[1:4]]
            v.append(vertex)
        if parts[0] == 'vt':
            texcoord = [float(i) for i in parts[1:3]]
            vt.append(texcoord)
        if parts[0] == 'f':
            vertex_index = [int(i.split('/')[0]) for i in parts[1:4]]
            v_idx.append(vertex_index)
            texcoord_index = [int(i.split('/')[1]) for i in parts[1:4]]
            vt_idx.append(texcoord_index)
    v = np.array(v)
    vt = np.array(vt)
    v_idx = np.array(v_idx)
    vt_idx = np.array(vt_idx)
    return v,vt,v_idx,vt_idx


def readObj(file_path):
    obj_file = open(file_path,'r')
    contents = obj_file.readlines()
    vertices, texcoords, faces = [],[],[]
    for line in contents:
        parts = line.split(' ')
        if parts[0] == 'v':
            vertex = [float(i) for i in parts[1:4]]
            vertices.append(vertex)
        if parts[0] == 'vt':
            texcoord = [float(i) for i in parts[1:3]]
            texcoords.append(texcoord)
        if parts[0] == 'f':
            face = [int(i.split('/')[0]) for i in parts[1:4]]
            faces.append(face)
    vertices = np.array(vertices)
    texcoords = np.array(texcoords)
    faces = np.array(faces)
    return vertices, texcoords, faces


def R_axis_angle(axis, angle):
    """Generate the rotation matrix from the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        c = cos(angle); s = sin(angle); C = 1-c
        xs = x*s;   ys = y*s;   zs = z*s
        xC = x*C;   yC = y*C;   zC = z*C
        xyC = x*yC; yzC = y*zC; zxC = z*xC
        [ x*xC+c   xyC-zs   zxC+ys ]
        [ xyC+zs   y*yC+c   yzC-xs ]
        [ zxC-ys   yzC+xs   z*zC+c ]

    @param axis:    The 3D rotation axis.
    @type axis:     numpy array, len 3
    @param angle:   The rotation angle.
    @type angle:    float
    """
    # Trig factors.
    ca = math.cos(angle)
    sa = math.sin(angle)
    C = 1 - ca

    # Depack the axis.
    x, y, z = axis

    # Multiplications (to remove duplicate calculations).
    xs = x*sa
    ys = y*sa
    zs = z*sa
    xC = x*C
    yC = y*C
    zC = z*C
    xyC = x*yC
    yzC = y*zC
    zxC = z*xC

    matrix = np.zeros((3,3))
    # Update the rotation matrix.
    matrix[0, 0] = x*xC + ca
    matrix[0, 1] = xyC - zs
    matrix[0, 2] = zxC + ys
    matrix[1, 0] = xyC + zs
    matrix[1, 1] = y*yC + ca
    matrix[1, 2] = yzC - xs
    matrix[2, 0] = zxC - ys
    matrix[2, 1] = yzC + xs
    matrix[2, 2] = z*zC + ca
    return matrix


def getScaleCenterfromKp(keypoints, kp_type, height_pixels = 150.):
    min_pt = np.amin(keypoints,0)
    max_pt = np.amax(keypoints,0)
    person_height = np.linalg.norm(max_pt - min_pt)
    if person_height == 0:
        print('bad!')
        import ipdb
        ipdb.set_trace()
    if kp_type == 'aug_joints_v2':
        center = keypoints[0] #pelvis
    else:
        raise NotImplementedError('joints type not supported!')

    scale = height_pixels / person_height
    return scale, center

def resize_img(img, scale_factor):
    new_size = (np.floor(np.array(img.shape[0:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [
        new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])
    ]
    return new_img, actual_factor

def scaleCrop(image, keypoints, scale, center, img_size=256):
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    image_scaled, scale_factors = resize_img(image, scale)
    # Swap so it's [x, y]
    scale_factors = np.array([scale_factors[1], scale_factors[0]])
    center_scaled = np.round(center * scale_factors).astype(np.int)
    keypoints_scaled = keypoints * scale_factors.reshape(1,-1)

    margin = int(img_size / 2)
    center_pad = center_scaled + margin
    keypoints_pad = keypoints_scaled + margin
    start_pt = center_pad - margin
    end_pt = center_pad + margin
    keypoints_crop = keypoints_pad - start_pt
    if len(image.shape) == 3: # rgb
        image_pad = np.pad(image_scaled, ((margin, ), (margin, ), (0, )), mode='edge')
        img_crop = image_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
    else: # mask
        image_pad = np.pad(image_scaled, ((margin, ), (margin, )), mode='edge')
        img_crop = image_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0]]
    proc_param = {
        'scale': scale,
        'start_pt': start_pt,
        'end_pt': end_pt,
        'img_size': img_size
    }
    return img_crop, keypoints_crop, proc_param