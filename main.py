import glob
import sys
import time
import haversine as hs
from haversine import Unit
import cv2
import numpy as np
from Matcher import match_image
import math
import os
import csv


distCoeffs = np.array([-0.11143032409169303,
                              0.0874541163945344,
                              0,
                              -8.486542026836708e-06,
                              0.00014540126479700144]
                      , dtype = "double")
# distCoeffs = np.zeros((5, 1))

intrisicMat = np.array([[1174.6973334747497,0,1001.7220922936314],
                        [0,1186.5886209150665,765.2708767726933],
                        [0,0,1]]
                       , dtype = "double")


def solve_image_pairs():
    if method == 'sift':
        H_found, correct_matched_map, correct_matched_img = match_image.sift_match(map, img)
    elif method == 'spp':
        H_found, correct_matched_map, correct_matched_img = match_image.spp_match(map_path, img_path)
    elif method == 'd2net':
        H_found, correct_matched_map, correct_matched_img = match_image.d2net_match(map_path, img_path)
    else:
        print("Wrong method instruct")
    return H_found, correct_matched_map, correct_matched_img


def direct_match_solve():
    h, w, _ = (cv2.imread(img_path)).shape
    gt_map_lon_res = (float(map_path.split("@")[4]) - float(map_path.split("@")[2])) / map.shape[1]
    gt_map_lat_res = (float(map_path.split("@")[5]) - float(map_path.split("@")[3])) / map.shape[0]

    LT_lon = 120.42114259488751
    LT_lat = 36.604504047017464
    RB_lon = 120.4568481612987
    RB_lat = 36.586863027841225

    gt_img_lon, gt_img_lat = (float(img_path.split("@")[1]) - LT_lon) / gt_map_lon_res, (
                float(img_path.split("@")[2]) - LT_lat) / gt_map_lat_res
    gt_map_lon, gt_map_lat = (float(map_path.split("@")[2]) - LT_lon) / gt_map_lon_res, (
                float(map_path.split("@")[3]) - LT_lat) / gt_map_lat_res

    H_found, map_points_3D, img_points_2D = solve_image_pairs()
    H_found_inv = np.linalg.inv(H_found)
    H_found_inv = H_found_inv / H_found_inv[-1, -1]
    # H_found_inv = H_found
    # mind the index of the first dimension and the second dimension
    [img_cx, img_cy, _] = np.dot(H_found_inv, np.array([w/2.0, h/2.0, 1]))
    img_c = np.array([img_cy, img_cx])
    (es_img_lat, es_img_lon) = img_c + np.array([gt_map_lat, gt_map_lon])
    dist_error = math.sqrt((gt_img_lat - es_img_lat) ** 2 + (gt_img_lon - es_img_lon) ** 2) * 0.381
    dir_sub_path = img_path.replace(img_path.split('\\')[-1], '')
    if not os.path.exists(dir_sub_path + '\\match_DM_{}.jpg'.format(dist_error)):
        os.rename(dir_sub_path + '\\match.jpg', dir_sub_path + '\\match_DM_{}.jpg'.format(dist_error))

    print("gt_img_lat: ", gt_img_lat, "es_img_lat: ", es_img_lat)
    print("gt_img_lon: ", gt_img_lon, "es_img_lon: ", es_img_lon)
    print("localization error is : {} m".format(dist_error))

    return dist_error


def PnP_solve():
    gt_map_lon_res = (float(map_path.split("@")[4])-float(map_path.split("@")[2])) / map.shape[1]
    gt_map_lat_res = (float(map_path.split("@")[5])-float(map_path.split("@")[3])) / map.shape[0]

    # longitude and latitude for the whole map
    LT_lon = 120.42114259488751
    LT_lat = 36.604504047017464
    RB_lon = 120.4568481612987
    RB_lat = 36.586863027841225

    gt_img_lon, gt_img_lat = (float(img_path.split("@")[1])-LT_lon)/gt_map_lon_res, (float(img_path.split("@")[2])-LT_lat)/gt_map_lat_res
    gt_map_lon, gt_map_lat = (float(map_path.split("@")[2])-LT_lon)/gt_map_lon_res, (float(map_path.split("@")[3])-LT_lat)/gt_map_lat_res

    _, map_points_3D, img_points_2D = solve_image_pairs()
    img_points_2D = np.array(img_points_2D, dtype = "double").squeeze()
    map_points_3D = np.array(map_points_3D, dtype = "double").squeeze()
    map_points_3D = map_points_3D + np.array([gt_map_lat, gt_map_lon])

    map_points_3D = np.concatenate((map_points_3D, np.zeros((map_points_3D.shape[0], 1))), axis=1)

    # success, vector_rotation, vector_translation, _ = cv2.solvePnPRansac(map_points_3D, img_points_2D, intrisicMat, distCoeffs)
    # flag=cv2.SOLVEPNP_EPNP might be wrong
    success, vector_rotation, vector_translation = cv2.solvePnP(map_points_3D, img_points_2D, intrisicMat, distCoeffs, flags=cv2.SOLVEPNP_EPNP)
    # https://blog.csdn.net/cocoaqin/article/details/77848588
    R, _ = cv2.Rodrigues(vector_rotation)
    inv_vector_translation = np.dot(-np.linalg.inv(R), vector_translation)

    es_img_lon, es_img_lat = inv_vector_translation[1], inv_vector_translation[0]
    loc_gt = (gt_img_lat, gt_img_lon)
    loc_es = (es_img_lat, es_img_lon)
    # dist_error = hs.haversine(loc_gt, loc_es, unit = Unit.METERS)
    dist_error = math.sqrt((gt_img_lat - es_img_lat[0]) ** 2 + (gt_img_lon - es_img_lon[0]) ** 2) * 0.381

    dir_sub_path = img_path.replace(img_path.split('\\')[-1], '')
    if not os.path.exists(dir_sub_path + '\\match_PnP_{}.jpg'.format(dist_error)):
        os.rename(dir_sub_path + '\\match.jpg', dir_sub_path + '\\match_PnP_{}.jpg'.format(dist_error))

    print("gt_img_lat: ", gt_img_lat, "es_img_lat: ", es_img_lat[0])
    print("gt_img_lon: ", gt_img_lon, "es_img_lon: ", es_img_lon[0])
    print("localization error is : {} m".format(dist_error))

    return dist_error


if __name__ == '__main__':
    dir_path = './data/sample/*'
    if os.path.exists('./data/result.csv'):
        os.remove('./data/result.csv')

    for dir_sub_path in glob.glob(dir_path):
        img_path, map_path = glob.glob(dir_sub_path+'\\*.png')
        img = cv2.imread(img_path)
        map = cv2.imread(map_path)
        method = 'd2net'
        start_time = time.time()
        print("\n=====PnP method=====")
        PnP_dist_error = PnP_solve()
        PnP_runtime = time.time() - start_time
        print("PnP running time is : {} s".format(PnP_runtime))
        print("\n=====direct align method=====")
        DrM_dist_error = direct_match_solve()
        DrM_runtime = time.time() - start_time - PnP_runtime
        print("direct align running time is : {} s".format(DrM_runtime))

        with open('./data/result.csv', 'a', encoding = 'UTF8', newline = '') as f:
            writer = csv.writer(f)
            line = [img_path, PnP_runtime, PnP_dist_error, DrM_runtime, DrM_dist_error]
            writer.writerow(line)