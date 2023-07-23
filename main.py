import time
import haversine as hs
from haversine import Unit
import cv2
import numpy as np
from Matcher import match_image
import math


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
    gt_img_lon, gt_img_lat = float(img_path.split("@")[2]), float(img_path.split("@")[3])
    gt_map_lon, gt_map_lat = float(map_path.split("@")[4]), float(map_path.split("@")[3])
    gt_map_lon_res = (float(map_path.split("@")[4]) - float(map_path.split("@")[2])) / map.shape[1]
    gt_map_lat_res = (float(map_path.split("@")[3]) - float(map_path.split("@")[5])) / map.shape[0]

    H_found, map_points_3D, img_points_2D = solve_image_pairs()
    H_found_inv = np.linalg.inv(H_found)
    [img_cy, img_cx, _] = np.dot(H_found_inv, np.array([h/2.0, w/2.0, 1]))
    img_c = np.array([img_cx, img_cy])
    es_img = - img_c * np.array([gt_map_lat_res, gt_map_lon_res]) + np.array([gt_map_lat, gt_map_lon])
    loc_gt = (gt_img_lat, gt_img_lon)
    loc_es = (es_img[0], es_img[1])
    dist_error = hs.haversine(loc_gt, loc_es, unit = Unit.METERS)

    print("gt_img_lat: {}, es_img_lat: {}",format(gt_img_lat, es_img[0]))
    print("gt_img_lon: {}, es_img_lon: {}",format(gt_img_lon, es_img[1]))
    print("localization error is : {} m".format(dist_error))


def PnP_solve():
    gt_img_lon, gt_img_lat = float(img_path.split("@")[2]), float(img_path.split("@")[3])
    gt_map_lon, gt_map_lat = float(map_path.split("@")[4]), float(map_path.split("@")[3])
    gt_map_lon_res = (float(map_path.split("@")[4])-float(map_path.split("@")[2])) / map.shape[1]
    gt_map_lat_res = (float(map_path.split("@")[3])-float(map_path.split("@")[5])) / map.shape[0]

    _, map_points_3D, img_points_2D = solve_image_pairs()
    img_points_2D = np.array(img_points_2D, dtype = "double").squeeze()
    map_points_3D = np.array(map_points_3D, dtype = "double").squeeze()
    map_points_3D = - map_points_3D * np.array([gt_map_lat_res, gt_map_lon_res]) + np.array([gt_map_lat, gt_map_lon])

    map_points_3D = np.concatenate((map_points_3D, np.zeros((map_points_3D.shape[0], 1))), axis=1)

    # success, vector_rotation, vector_translation, _ = cv2.solvePnPRansac(map_points_3D, img_points_2D, intrisicMat, distCoeffs)
    success, vector_rotation, vector_translation = cv2.solvePnP(map_points_3D, img_points_2D, intrisicMat, distCoeffs, flags=cv2.SOLVEPNP_EPNP)
    # https://blog.csdn.net/cocoaqin/article/details/77848588
    R, _ = cv2.Rodrigues(vector_rotation)
    inv_vector_translation = np.dot(-np.linalg.inv(R), vector_translation)

    es_img_lon, es_img_lat = inv_vector_translation[1], inv_vector_translation[0]
    loc_gt = (gt_img_lat, gt_img_lon)
    loc_es = (es_img_lat, es_img_lon)
    dist_error = hs.haversine(loc_gt, loc_es, unit = Unit.METERS)
    # dist_error = math.sqrt((gt_img_lat - es_img_lat[0]) ** 2 + (gt_img_lon - es_img_lon[0]) ** 2) * 0.381

    print("gt_img_lat: {}, es_img_lat: {}", format(gt_img_lat, es_img_lat[0].numel))
    print("gt_img_lon: {}, es_img_lon: {}", format(gt_img_lon, es_img_lon[0].numel))
    print("localization error is : {} m".format(dist_error))


if __name__ == '__main__':
    map_path = '.\\data\\sample1\\@map@120.44313670881267@36.602350602293804@120.44528247602489@36.600627846514875@.png'
    img_path = '.\\data\\sample1\\@1679113316349@120.444@36.60093166666667@.png'
    # map_path = '.\\data\\sample1\\@map@0@0@500@500@.png'
    # img_path = '.\\data\\sample1\\@1679113316349@298.83858@411.82146781@.png'
    img = cv2.imread(img_path)
    map = cv2.imread(map_path)
    method = 'd2net'
    start_time = time.time()
    print("=====PnP method=====", end = '\n')
    PnP_solve()
    print("=====direct align method=====")
    direct_match_solve()
    print("running time is : {} s".format(time.time()-start_time))