import cv2
import numpy as np
from Matcher import match_image


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


def main():
    gt_img_lon, gt_img_lat = float(img_path.split("@")[2]), float(img_path.split("@")[3])
    gt_map_lon, gt_map_lat = float(map_path.split("@")[4]), float(map_path.split("@")[3])
    gt_map_lon_res = (float(map_path.split("@")[4])-float(map_path.split("@")[2])) / map.shape[1]
    gt_map_lat_res = (float(map_path.split("@")[3])-float(map_path.split("@")[5])) / map.shape[0]

    _, map_points_3D, img_points_2D = solve_image_pairs()
    img_points_2D = np.array(img_points_2D, dtype = "double").squeeze()
    map_points_3D = np.array(map_points_3D, dtype = "double").squeeze()
    map_points_3D = - map_points_3D * np.array([gt_map_lat_res, gt_map_lon_res]) + np.array([gt_map_lat, gt_map_lon])

    map_points_3D = np.concatenate((map_points_3D, np.zeros((map_points_3D.shape[0], 1))), axis=1)

    success, vector_rotation, vector_translation = cv2.solvePnP(map_points_3D, img_points_2D, intrisicMat,
                                                                distCoeffs, flags = 0)
    es_img_lon, es_img_lat = vector_translation[1], vector_translation[0]

    print(gt_img_lat, es_img_lat)
    print(gt_img_lon, es_img_lon)
    print("EMD")


if __name__ == '__main__':
    map_path = '.\\data\\sample1\\@map@120.4423996416947@36.60208593028245@120.44431458039664@36.60122394650841@.png'
    img_path = '.\\data\\sample1\\@1679113316349@120.444@36.60093166666667@.png'
    img = cv2.imread(img_path)
    map = cv2.imread(map_path)
    method = 'd2net'
    main()