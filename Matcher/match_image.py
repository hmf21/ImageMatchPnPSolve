import time
import cv2
import numpy as np
import sys

import warnings
warnings.filterwarnings("ignore")

import Matcher.superpoint.demo_superpoint as superpoint
import Matcher.d2net.d2net as d2net


def sift_match(map, img):
    h, w, _ = img.shape

    if map.shape[0] == 3:
        map = np.swapaxes(map, 0, 2)
        map = np.swapaxes(map, 0, 1)
        map = (map * 255).astype('uint8')

    if img.shape[0] == 3:
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 0, 1)
        img = (img * 255).astype('uint8')

    map_gray = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(map_gray, None)
    kp2, des2 = sift.detectAndCompute(img_gray, None)

    if (len(kp1) >= 10) and (len(kp2) >= 10):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append(m)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        if (src_pts.size <= 5) or (dst_pts.size <= 5):
            H_found = np.eye(3)
        else:
            try:
                H_found, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                correct_matched_map = [src_pts[i] for i in range(len(good)) if mask[i]]
                correct_matched_img = [dst_pts[i] for i in range(len(good)) if mask[i]]
            except cv2.error:
                H_found = np.eye(3)
                correct_matched_map = None
                correct_matched_img = None
        if H_found is None:
            H_found = np.eye(3)
            correct_matched_map = None
            correct_matched_img = None
    else:
        H_found = np.eye(3)
        correct_matched_map = None
        correct_matched_img = None
    return H_found, correct_matched_map, correct_matched_img


def d2net_match(img1_path, img2_path, idx=0, save_path=None):
    [image_1_keypoints, image_1_scores, image_1_descriptors] = d2net.d2net_extractor(img1_path)
    [image_2_keypoints, image_2_scores, image_2_descriptors] = d2net.d2net_extractor(img2_path)

    image_1_kpts = []
    for keypoint in image_1_keypoints:
        image_1_kpts.append(cv2.KeyPoint(x=keypoint[0], y=keypoint[1], size=1))

    image_2_kpts = []
    for keypoint in image_2_keypoints:
        image_2_kpts.append(cv2.KeyPoint(x=keypoint[0], y=keypoint[1], size=1))

    # # # brute-force match
    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)
    # matches = bf.match(image_1_descriptors, image_2_descriptors)
    # matches = sorted(matches, key = lambda x: (x.distance) * (1))
    # correct_matched_map = [image_1_kpts[i.queryIdx].pt for i in matches[:30]]
    # correct_matched_img = [image_2_kpts[i.trainIdx].pt for i in matches[:30]]
    # return None, correct_matched_map, correct_matched_img

    if (len(image_1_kpts) >= 10) and (len(image_2_kpts) >= 10):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Use D2-Net dep
        matches = flann.knnMatch(image_1_descriptors, image_2_descriptors, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.95 * n.distance:
                good.append(m)

        src_pts = np.float32([image_1_kpts[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        src_kpts = []
        for keypoint in src_pts:
            src_kpts.append(cv2.KeyPoint(x=keypoint[0, 0], y=keypoint[0, 1], size=1))
        dst_pts = np.float32([image_2_kpts[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_kpts = []
        for keypoint in dst_pts:
            dst_kpts.append(cv2.KeyPoint(x=keypoint[0, 0], y=keypoint[0, 1], size=1))

        if (src_pts.size <= 3) or (dst_pts.size <= 3):
            H_found = np.eye(3)
        else:
            H_found, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            correct_matched_map = [src_pts[i] for i in range(len(good)) if mask[i]]
            correct_matched_img = [dst_pts[i] for i in range(len(good)) if mask[i]]
        if H_found is None:
            H_found = np.eye(3)
            correct_matched_map = None
            correct_matched_img = None
    else:
        H_found = np.eye(3)
        correct_matched_map = None
        correct_matched_img = None
    return H_found, correct_matched_map, correct_matched_img


def spp_match(img1_path, img2_path):
    # parameters setup
    weights_path = './Matcher/superpoint/superpoint_v1.pth'
    nms_dist = 4
    conf_thresh = 0.015
    nn_thresh = 0.7
    cuda = True
    fe = superpoint.SuperPointFrontend(weights_path, nms_dist, conf_thresh, nn_thresh, cuda)

    interp = cv2.INTER_AREA
    rz_h_size = 500
    image_1 = cv2.imread(img1_path, 0)
    img_1_h = image_1.shape[0]
    rz_im_1_shape = (image_1.shape[0] / image_1.shape[1]) * 1.0
    image_1_color = cv2.imread(img1_path)
    image_1 = cv2.resize(image_1, (int(rz_h_size / rz_im_1_shape), rz_h_size), interpolation = interp)
    image_1_color = cv2.resize(image_1_color, (int(rz_h_size / rz_im_1_shape), rz_h_size), interpolation = interp)
    image_1 = (image_1.astype('float32') / 255.)
    pts1, image_1_descriptors, heatmap1 = fe.run(image_1)
    image_1_descriptors = np.swapaxes(image_1_descriptors, 0, 1)
    image_1_kpts = []
    pts1 = np.swapaxes(pts1, 0, 1)
    for keypoint in pts1:
        image_1_kpts.append(cv2.KeyPoint(x=keypoint[0], y=keypoint[1], size=1))

    image_2 = cv2.imread(img2_path, 0)
    rz_im_2_shape = (image_1.shape[0] / image_1.shape[1]) * 1.0
    image_2_color = cv2.imread(img2_path)
    img_2_h = image_2.shape[0]
    image_2 = cv2.resize(image_2, (int(rz_h_size / rz_im_2_shape), rz_h_size), interpolation = interp)
    image_2_color = cv2.resize(image_2_color, (int(rz_h_size / rz_im_2_shape), rz_h_size), interpolation = interp)
    image_2 = (image_2.astype('float32') / 255.)
    pts2, image_2_descriptors, heatmap2 = fe.run(image_2)
    time_check_2 = time.time()
    image_2_descriptors = np.swapaxes(image_2_descriptors, 0, 1)
    image_2_kpts = []
    pts2 = np.swapaxes(pts2, 0, 1)
    for keypoint in pts2:
        image_2_kpts.append(cv2.KeyPoint(x=keypoint[0], y=keypoint[1], size=1))

    if (len(image_1_kpts) >= 10) and (len(image_2_kpts) >= 10):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # 使用k近邻匹配的方式，所以这里的返回值是2，
        matches = flann.knnMatch(image_1_descriptors, image_2_descriptors, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        if len(good) > 10:
            src_pts = np.float32([image_1_kpts[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            src_kpts = []
            for keypoint in src_pts:
                src_kpts.append(cv2.KeyPoint(x=keypoint[0, 0], y=keypoint[0, 1], size=1))
            dst_pts = np.float32([image_2_kpts[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_kpts = []
            for keypoint in dst_pts:
                dst_kpts.append(cv2.KeyPoint(x=keypoint[0, 0], y=keypoint[0, 1], size=1))

            if (src_pts.size <= 3) or (dst_pts.size <= 3):
                H_found = np.eye(3)
                correct_matched_map = None
                correct_matched_img = None
            else:
                H_found, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
                correct_matched_map = [src_pts[i] * img_1_h / rz_h_size for i in range(len(good)) if mask[i]]
                correct_matched_img = [dst_pts[i] * img_2_h / rz_h_size for i in range(len(good)) if mask[i]]
        else:
            H_found = np.eye(3)
            correct_matched_map = None
            correct_matched_img = None
    else:
        H_found = np.eye(3)
        correct_matched_map = None
        correct_matched_img = None
    return H_found, correct_matched_map, correct_matched_img