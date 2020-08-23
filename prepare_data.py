import os
import os.path as osp
from glob import glob
import pickle
import numpy as np
import cv2
from utils import *

def check_match(verts, j3d, thre=0.05):
    j3d_regressor = np.load('j3d_regressor.npy')
    j3d_regress = np.dot(j3d_regressor,verts)
    dist = np.sqrt(np.sum((j3d[0] - j3d_regress[0])**2))
    if dist < thre: 
        return True
    else:
        return False

def get_rot_mat(j3d, a_id=0, b_id=10, c_id=13):
    # a: pelvis, b: lshoulder, c: rshoulder
    v_norm = np.cross(j3d[b_id]-j3d[a_id], j3d[c_id]-j3d[a_id])
    v_norm = np.array([v_norm[0],0,v_norm[2]])
    gt_norm = np.array([1.,0,0])
    v_norm = v_norm / np.linalg.norm(v_norm)
    gt_norm = gt_norm / np.linalg.norm(gt_norm)
    angle = np.arccos(np.dot(v_norm, gt_norm))
    if v_norm[2] < 0:
        angle = 2 * np.pi - angle
    axis = np.array([0,1.,0])
    axis_angle = axis * angle
    rot_mat = R_axis_angle(axis, angle)
    inv_rot_mat = rot_mat.T
    return rot_mat, inv_rot_mat

def parse_seg_file(seg_file_path):
    part_names = ['head', 'arm', 'shirt', 'pant', 'leg']
    seg_file = open(seg_file_path,'r')
    seg_file_content = seg_file.readlines()
    dir_dict = {}
    for line in seg_file_content:
        line = line.strip('.\n')
        # new folder
        if line[0] != '\t':
            nba_dir = line
            dir_key = nba_dir
            if dir_key not in dir_dict:
                dir_dict[dir_key] = {}
        # new person
        elif line[1] != '\t':
            person_id = int(re.findall(r'\d+',line)[0])
            if person_id not in dir_dict[dir_key]:
                dir_dict[dir_key][person_id] = []
        else:
            obj_name = line.strip('\t')
            body_part_name = obj_name.split('_')[1]
            if body_part_name in part_names:
                dir_dict[dir_key][person_id].append(obj_name)
    return dir_dict

def saveDataDict(mesh_dir, pose_dir, images_dir, write_dir, player, dir_type, joint_type):
    os.makedirs(write_dir, exist_ok=True)
    # read rest pose
    rest_j3d,_,_ = readObj(osp.join(pose_dir, 'rest_pose_data/players/{}.obj'.format(joint_type)))
    j3d_blend = np.load(osp.join(pose_dir, 'rest_pose_data/players/{}.npy'.format(joint_type)))
    # parse seg file
    if '/' in player:
        seg_file_path = osp.join(mesh_dir, 'seg_release', 'seg_{}_{}_{}.txt'.format(player.split('/')[0], player.split('/')[1], dir_type))
    else:
        seg_file_path = osp.join(mesh_dir, 'seg_release', 'seg_{}_{}.txt'.format(player, dir_type))
    dir_dict = parse_seg_file(seg_file_path)

    data_dict = {
        'head_verts': [],
        'arm_verts': [],
        'shirt_verts': [],
        'pant_verts': [],
        'leg_verts': [],
        'shoes_verts': [],
        'human_verts': [],
        'j3d': [],
        'player': [],
        'dir_type': [],
        'nba_dir': [],
        'person_id': [],
    }
    dir_num = len(dir_dict.items())
    for idx, (nba_dir, human_dict) in enumerate(dir_dict.items()):
        print(player, idx, dir_num, nba_dir)
        # load nba dir meta data
        img_path = osp.join(images_dir, player, dir_type, '{}.png'.format(nba_dir))
        if not osp.exists(img_path):
            print('image not exist!')
            continue
        proj_mat_math = osp.join(pose_dir, player, dir_type, nba_dir, 'proj_mat.npy')
        if not osp.exists(proj_mat_math):
            print('proj mat not exist!')
            continue
        img = cv2.imread(img_path)
        proj_mat = np.load(proj_mat_math)
        # iterate over person in nba dir
        for person_id, parts_list in human_dict.items():
            trans_mat_path = osp.join(pose_dir,player,dir_type,nba_dir,'players','{}_person_v2_transform_v2.npy'.format(person_id))
            if not osp.exists(trans_mat_path):
                print('trans mat not exists!')
                continue
            mesh_path = osp.join(mesh_dir,player,'resampled',dir_type,nba_dir,'players','{}_person_simple.obj'.format(person_id))
            if not osp.exists(mesh_path):
                print('mesh not exists!')
                continue
            verts,_,_ = readObj(mesh_path)
            # forward kinematics to get j3d
            struc_buf = np.load(trans_mat_path)
            per_joint_transform = struc_buf[j3d_blend]
            deformed_j3d = rest_j3d * 100. # m to cm
            ones = np.ones((deformed_j3d.shape[0],1))
            deformed_j3d = np.expand_dims(np.concatenate((deformed_j3d,ones),1),-1)
            deformed_j3d = np.matmul(per_joint_transform,deformed_j3d)
            deformed_j3d = deformed_j3d.reshape(deformed_j3d.shape[0],deformed_j3d.shape[1])
            deformed_j3d *= 0.01 # cm to m
            # check if j3d and verts match to the same person
            if not check_match(verts, deformed_j3d):
                print('j3d and verts not matched!')
                continue
            # project joints
            joints = deformed_j3d.copy() * 100. # m to cm
            ones = np.ones((joints.shape[0],1))
            joints_homo = np.concatenate((joints,ones),-1)
            clip_coord = np.dot(proj_mat, joints_homo.T).T
            ndc_coord = clip_coord[:,0:3] / clip_coord[:,3].reshape(-1,1)
            raster_coord = np.zeros((ndc_coord.shape[0],2))
            raster_coord[:,0] = (ndc_coord[:,0]+1.)*0.5*(1280-1)
            raster_coord[:,1] = (1-(ndc_coord[:,1]+1.)*0.5)*(800-1)
            keypoints = raster_coord.copy()
            # boundary test
            min_pt,max_pt=np.amin(keypoints,0), np.amax(keypoints,0)
            if min_pt[0] < 0 or max_pt[0] > img.shape[1] or min_pt[1] < 0 or max_pt[1] > img.shape[0]:
                print('outside image!')
                continue
            
            # save data
            data_dict['player'].append(player)
            data_dict['dir_type'].append(dir_type)
            data_dict['nba_dir'].append(nba_dir)
            data_dict['person_id'].append(person_id)

            write_img_dir = osp.join(write_dir, player, 'images', dir_type)
            os.makedirs(write_img_dir, exist_ok=True)
            write_img_path = osp.join(write_img_dir, '{}_{:02d}.png'.format(nba_dir, person_id))

            # scale image
            if joint_type == 'aug_joints_v2':
                jidx = 16
            else:
                raise NotImplementedError('joints type not supported!')
            scale, center = getScaleCenterfromKp(keypoints[0:jidx], kp_type=joint_type, height_pixels = 150.)
            img_crop, keypoints_crop, proc_param = scaleCrop(img, keypoints, scale, center)
            cv2.imwrite(write_img_path, img_crop)

            # move j3d to origin, rotate so that facing x axis
            rot_mat,inv_rot_mat = get_rot_mat(deformed_j3d)
            offset = deformed_j3d[0:1].copy()
            deformed_j3d -= offset
            deformed_j3d = np.dot(rot_mat, deformed_j3d.T).T
            data_dict['j3d'].append(deformed_j3d)
            
            # save single body part
            for part_name in parts_list:
                body_part_obj_path = osp.join(mesh_dir,player,'resampled',dir_type,nba_dir,'player_parts','{}.obj'.format(part_name))
                body_part_verts,_,_ = readObj(body_part_obj_path)
                body_part_verts -= offset
                body_part_verts = np.dot(rot_mat, body_part_verts.T).T
                body_part_name = part_name.split('_')[-1]
                data_dict['{}_verts'.format(body_part_name)].append(body_part_verts)
            # save shoes
            body_part_obj_path = osp.join(mesh_dir,player,'resampled',dir_type,nba_dir,'player_parts','{}_shoes.obj'.format(person_id))
            body_part_verts,_,_ = readObj(body_part_obj_path)
            body_part_verts -= offset
            body_part_verts = np.dot(rot_mat, body_part_verts.T).T
            data_dict['shoes_verts'].append(body_part_verts)
            # save human
            human_verts,_,_ = readObj(mesh_path)
            human_verts -= offset
            human_verts = np.dot(rot_mat, human_verts.T).T
            data_dict['human_verts'].append(human_verts)
    
    if '/' in player:
        pkl_path = osp.join(write_dir, player, '{}_{}_{}.pkl'.format(player.split('/')[0], player.split('/')[-1], dir_type))
    else:
        pkl_path =  osp.join(write_dir, player, '{}_{}.pkl'.format(player, dir_type))
    pickle.dump(data_dict, open(pkl_path, 'wb'), protocol=2)


def main():
    root_dir = '/mnt/projects/lyzhu/nba/data/release'
    mesh_dir = osp.join(root_dir, 'mesh')
    pose_dir = osp.join(root_dir, 'pose')
    images_dir = osp.join(root_dir,'images')
    write_dir = osp.join(root_dir, 'training_data', 'mesh_release')
    
    player_list = [
        'alfred','chad', 'donell', 'erik', 'guy','jamaal','juwan',
        'kedrick','martin','nick','randall','zach','zack', 'lamond', 
        'cedric', 'dion', 'leo', 'lucas', 'brendan', 'oscar', 'barney',
        'allen', 'devin', 'darrell', 'bradley', 'glen', 'cory', 'tomas'
    ]
    dir_types = ['2ku', 'normal']
    joint_type = 'aug_joints_v2'
    for player in player_list:
        player = 'release/{}'.format(player)
        for dir_type in dir_types:
            saveDataDict(mesh_dir, pose_dir, images_dir, write_dir, player, dir_type, joint_type)


if __name__ == '__main__':
    main()
