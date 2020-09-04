import os
import os.path as osp
from glob import glob
import numpy as np
from shapely.geometry import LineString, Polygon
import cv2
import pickle
from imgaug.augmentables.segmaps import SegmentationMapOnImage


W, H = 28.65, 15.24

def inside_frame(points2d, height, width, margin=0):
    valid = np.logical_and(np.logical_and(points2d[:, 0] >= 0+margin, points2d[:, 0] < width-margin),
                           np.logical_and(points2d[:, 1] >= 0+margin, points2d[:, 1] < height-margin))
    points2d = points2d[valid, :]
    return points2d, valid

def make_field_circle(center, r, nn=30):
    """
    Returns points that lie on a circle on the ground
    :param r: radius
    :param nn: points per arc?
    :return: 3D points on a circle with y = 0
    """
    cx, cy, cz = center[0], center[1], center[2]
    d = 2 * np.pi * r
    n = int(nn * d)
    return [(cx+np.cos(2 * np.pi / n * x) * r, cy+0, cz+np.sin(2 * np.pi / n * x) * r) for x in range(0, n + 1)]

def get_field_points():
    # rectangles
    outer_rect = np.array([[-H/2., 0, -W/2.],
                            [-H/2., 0, W/2.],
                            [H/2., 0, W/2.],
                            [H/2., 0, -W/2.]])
    l_ft_rect = np.array([[-2.438, 0., -W/2.],
                         [ 2.438, 0., -W/2.],
                         [ 2.438, 0., -W/2.+5.79],
                         [-2.438, 0., -W/2.+5.79]])
    r_ft_rect = np.array([[-2.438, 0., W/2.-5.79],
                         [ 2.438, 0., W/2.-5.79],
                         [ 2.438, 0., W/2.],
                         [-2.438, 0., W/2.]])
    # lines
    mid_line = np.array([[H/2., 0.,0.],
                        [-H/2., 0., 0.]])
    ul_3pt_line = np.array([[H/2.-0.914, 0.,-W/2.],
                                [H/2.-0.914, 0., -W/2.+4.267]])
    bl_3pt_line = np.array([[-H/2.+0.914, 0.,-W/2.],
                                [-H/2.+0.914, 0., -W/2.+4.267]])
    ur_3pt_line = np.array([[H/2.-0.914, 0.,W/2.],
                                [H/2.-0.914, 0., W/2.-4.267]])
    br_3pt_line = np.array([[-H/2.+0.914, 0.,W/2.],
                                [-H/2.+0.914, 0., W/2.-4.267]])

    # circles
    central_circle = np.array(make_field_circle(center=(0,0,0), r=1.8288))

    l_ft_circle = np.array(make_field_circle(center=(0,0,-W/2.+5.7912), r=1.8288))
    index = l_ft_circle[:, 2] > (-W/2.+5.7912)
    l_ft_circle = l_ft_circle[index, :]

    r_ft_circle = np.array(make_field_circle(center=(0,0,W/2.-5.7912), r=1.8288))
    index = r_ft_circle[:, 2] < (W/2.-5.7912)
    r_ft_circle = r_ft_circle[index, :]

    l_restricted_circle = np.array(make_field_circle(center=(0,0,-W/2.+1.584), r=1.219))
    index = l_restricted_circle[:, 2] > (-W/2.+1.584)
    l_restricted_circle = l_restricted_circle[index, :]

    r_restricted_circle = np.array(make_field_circle(center=(0,0,W/2.-1.584), r=1.219))
    index = r_restricted_circle[:, 2] < (W/2.-1.584)
    r_restricted_circle = r_restricted_circle[index, :]

    l_3pt_circle = np.array(make_field_circle(center=(0,0,-W/2.+1.584), r=7.239))
    index = l_3pt_circle[:, 2] > (-W/2.+4.267)
    l_3pt_circle = l_3pt_circle[index, :]

    r_3pt_circle = np.array(make_field_circle(center=(0,0,W/2.-1.584), r=7.239))
    index = r_3pt_circle[:, 2] < (W/2.-4.267)
    r_3pt_circle = r_3pt_circle[index, :]

    return [outer_rect, l_ft_rect, r_ft_rect,
            mid_line, ul_3pt_line, bl_3pt_line,
            ur_3pt_line, br_3pt_line,
            central_circle, l_ft_circle, r_ft_circle,
            l_restricted_circle, r_restricted_circle,
            l_3pt_circle, r_3pt_circle]

def project_field_to_image(cam_params,h,w):

    field_list = get_field_points()

    field_points2d = []
    for i in range(len(field_list)):
        field_3d = field_list[i] * 100. # m to cm
        if cam_params.shape[0] == 15:
            rvec = cam_params[0:3].reshape(3,1)
            tvec = cam_params[3:6].reshape(3,1)
            mtx = cam_params[6:].reshape(3,3)
            cam_distortion = np.zeros(5)
            field_2d,_ = cv2.projectPoints(field_3d, rvec, tvec, mtx, cam_distortion)
            field_2d = field_2d.reshape(-1,2)
        elif cam_params.shape[0] == 16:
            proj_mat = cam_params.reshape(4,4)
            ones = np.ones((field_3d.shape[0],1))
            field_3d_homo = np.concatenate((field_3d,ones),-1)
            clip_coord = np.dot(proj_mat, field_3d_homo.T).T
            ndc_coord = clip_coord[:,0:3] / clip_coord[:,3:]
            raster_coord = np.zeros((ndc_coord.shape[0],2))
            raster_coord[:,0] = (ndc_coord[:,0]+1.)*0.5*(w-1)
            raster_coord[:,1] = (1-(ndc_coord[:,1]+1.)*0.5)*(h-1)
            field_2d, depth = raster_coord, ndc_coord[:,2:]
            # behind_points = (depth < 0).nonzero()[0]
            # field_2d[behind_points, :] *= -1

        field_points2d.append(field_2d)

    return field_points2d


def draw_field(cam_params, h, w, thickness=7):

    field_points2d = project_field_to_image(cam_params,h,w)
    # Check if the entities are 15
    assert len(field_points2d) == 15

    img_polygon = Polygon([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)])

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw the boxes
    for i in range(3):

        # And make a new image with the projected field
        linea = LineString([(field_points2d[i][0, :]),
                            (field_points2d[i][1, :])])

        lineb = LineString([(field_points2d[i][1, :]),
                            (field_points2d[i][2, :])])

        linec = LineString([(field_points2d[i][2, :]),
                            (field_points2d[i][3, :])])

        lined = LineString([(field_points2d[i][3, :]),
                            (field_points2d[i][0, :])])

        if i == 0:
            polygon0 = Polygon([(field_points2d[i][0, :]),
                                (field_points2d[i][1, :]),
                                (field_points2d[i][2, :]),
                                (field_points2d[i][3, :])])

            intersect0 = img_polygon.intersection(polygon0)
            if not intersect0.is_empty:
                pts = np.array(list(intersect0.exterior.coords), dtype=np.int32)
                pts = pts[:, :].reshape((-1, 1, 2))
                cv2.fillConvexPoly(mask, pts, (255, 255, 255))

        intersect0 = img_polygon.intersection(linea)
        if not intersect0.is_empty:
            pts = np.array(list(list(intersect0.coords)), dtype=np.int32)
            cv2.line(canvas, (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]), (255, 255, 255),thickness)

        intersect0 = img_polygon.intersection(lineb)
        if not intersect0.is_empty:
            pts = np.array(list(list(intersect0.coords)), dtype=np.int32)
            # if pts.shape[0] < 2:
            #     continue
            cv2.line(canvas, (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]), (255, 255, 255),thickness)

        intersect0 = img_polygon.intersection(linec)
        if not intersect0.is_empty:
            pts = np.array(list(list(intersect0.coords)), dtype=np.int32)
            # if pts.shape[0] == 2:
            cv2.line(canvas, (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]), (255, 255, 255),thickness)

        intersect0 = img_polygon.intersection(lined)
        if not intersect0.is_empty:
            pts = np.array(list(list(intersect0.coords)), dtype=np.int32)
            cv2.line(canvas, (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]), (255, 255, 255),thickness)

    # lines
    for i in range(3, 8):
        line1 = LineString([(field_points2d[i][0, :]),
                            (field_points2d[i][1, :])])

        intersect1 = img_polygon.intersection(line1)
        if not intersect1.is_empty:
            pts = np.array(list(list(intersect1.coords)), dtype=np.int32)
            # pts = pts[:, :].reshape((-1, 1, 2))
            # cv2.fillConvexPoly(canvas, pts, (255, 255, 255), )
            cv2.line(canvas, (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]), (255, 255, 255),thickness)

    # Circles
    for ii in range(8, 15):
        for i in range(field_points2d[ii].shape[0] - 1):
            line2 = LineString([(field_points2d[ii][i, :]),
                                (field_points2d[ii][i + 1, :])])
            intersect2 = img_polygon.intersection(line2)
            if not intersect2.is_empty:
                pts = np.array(list(list(intersect2.coords)), dtype=np.int32)
                # pts = pts[:, :].reshape((-1, 1, 2))
                # cv2.fillConvexPoly(canvas, pts, (255, 255, 255), )
                cv2.line(canvas, (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]), (255, 255, 255),thickness)

    return canvas[:, :, 0] / 255., mask[:, :, 0] / 255.

def create_dir(dir_name):
    if not osp.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def gen_line_map(root_dir, player_name, dir_type, h=800, w=1280):
    
    pose_dir_list = sorted(glob(osp.join(root_dir, 'pose', player_name, dir_type, '*')))
    write_dir = osp.join(root_dir, 'court_lines', player_name, dir_type)
    os.makedirs(write_dir, exist_ok=True)
    for idx, dir_name in enumerate(pose_dir_list):
        print(idx, len(pose_dir_list), dir_name)
        nba_dir = dir_name.split('/')[-1]
        img_path = osp.join(root_dir, 'images', player_name, dir_type, '{}.png'.format(nba_dir))
        if not osp.exists(img_path):
            print('image does not exist')
            continue
        proj_mat_path = osp.join(dir_name, 'proj_mat.npy')
        if not osp.exists(proj_mat_path):
            print('projection matrix does not exist')
            continue
        # need to check this, some images are close up animations
        trans_mat_paths = sorted(glob(osp.join(dir_name, 'players/*_person_v2_transform_v2.npy')))
        if len(trans_mat_paths) == 0:
            print('no valid players')
            continue
        proj_mat = np.load(proj_mat_path)
        cam_params = proj_mat.copy().ravel()
        canvas, mask = draw_field(cam_params, h, w)
        canvas = (canvas*255.).astype(np.uint8)
        write_path = osp.join(write_dir, img_path.split('/')[-1])
        cv2.imwrite(write_path, canvas)

def main():
    root_dir = '/mnt/projects/lyzhu/nba/data/release'
    player_list = [
        'alfred','chad', 'donell', 'erik', 'guy','jamaal','juwan',
        'kedrick','martin','nick','randall','zach','zack', 'lamond', 
        'cedric', 'dion', 'leo', 'lucas', 'brendan', 'oscar', 'barney',
        'allen', 'devin', 'darrell', 'bradley', 'glen', 'cory', 'tomas'
    ]
    dir_type_list = ['2ku', 'normal']
    
    for player_name in player_list:
        for dir_type in dir_type_list:
            new_player_name = 'release/{}'.format(player_name)
            gen_line_map(root_dir, new_player_name, dir_type)


if __name__ == '__main__':
    main()
