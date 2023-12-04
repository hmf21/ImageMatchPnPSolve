import cv2
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transform
from torch.autograd import Variable
import torch
import warnings
warnings.filterwarnings("ignore")

import Matcher.superpoint.demo_superpoint as superpoint
import Matcher.d2net.d2net as d2net
import deep_feat_VGG16.DeepLKBatch as dlk
from deep_feat_VGG16.config import *


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
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)
    matches = bf.match(image_1_descriptors, image_2_descriptors)
    matches = sorted(matches, key = lambda x: (x.distance) * (1))
    # image_1_gray = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)
    # image_2_gray = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)
    img3 = cv2.drawMatches(cv2.imread(img1_path).astype('uint8'), image_1_kpts,
                           cv2.imread(img2_path).astype('uint8'), image_2_kpts, matches[:20], None, flags = 2)
    dir_sub_path = img1_path.replace(img1_path.split('\\')[-1], '')
    if os.path.exists(dir_sub_path + '\\match.jpg'):
        os.remove(dir_sub_path + '\\match.jpg')
    cv2.imwrite(dir_sub_path + '\\match.jpg', img3)
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
            correct_matched_map = None
            correct_matched_img = None
        else:
            H_found, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # correct_matched_map = [src_pts[i] for i in range(len(good)) if mask[i]]
            correct_matched_map = src_pts
            # correct_matched_img = [dst_pts[i] for i in range(len(good)) if mask[i]]
            correct_matched_img = dst_pts
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
    rz_im_1_shape = (image_1.shape[0] / image_1.shape[1]) * 1.0
    image_1 = cv2.resize(image_1, (int(rz_h_size / rz_im_1_shape), rz_h_size), interpolation = interp)
    image_1 = (image_1.astype('float32') / 255.)
    pts1, image_1_descriptors, heatmap1 = fe.run(image_1)
    image_1_descriptors = np.swapaxes(image_1_descriptors, 0, 1)
    image_1_kpts = []
    pts1 = np.swapaxes(pts1, 0, 1)
    for keypoint in pts1:
        image_1_kpts.append(cv2.KeyPoint(x=keypoint[0], y=keypoint[1], size=1))

    image_2 = cv2.imread(img2_path, 0)
    rz_im_2_shape = (image_1.shape[0] / image_1.shape[1]) * 1.0
    image_2 = cv2.resize(image_2, (int(rz_h_size / rz_im_2_shape), rz_h_size), interpolation = interp)
    image_2 = (image_2.astype('float32') / 255.)
    pts2, image_2_descriptors, heatmap2 = fe.run(image_2)
    image_2_descriptors = np.swapaxes(image_2_descriptors, 0, 1)
    image_2_kpts = []
    pts2 = np.swapaxes(pts2, 0, 1)
    for keypoint in pts2:
        image_2_kpts.append(cv2.KeyPoint(x=keypoint[0], y=keypoint[1], size=1))

    # # # brute-force match
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)
    matches = bf.match(image_1_descriptors, image_2_descriptors)
    matches = sorted(matches, key = lambda x: (x.distance) * (1))

    img3 = cv2.drawMatches(cv2.imread(img1_path).astype('uint8'), image_1_kpts,
                           cv2.imread(img2_path).astype('uint8'), image_2_kpts, matches[:20], None, flags = 2)
    dir_sub_path = img1_path.replace(img1_path.split('\\')[-1], '')
    if os.path.exists(dir_sub_path + '\\match.jpg'):
        os.remove(dir_sub_path + '\\match.jpg')
    cv2.imwrite(dir_sub_path + '\\match.jpg', img3)

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
            correct_matched_map = None
            correct_matched_img = None
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


def dlk_match(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # accelarate
    scaled_im_height = 100
    scaled_im_weight = round(img1.shape[1] * scaled_im_height / img1.shape[0])
    img1_PIL = Image.fromarray(img1)
    img1_PIL_rz = img1_PIL.resize((scaled_im_weight, scaled_im_height))
    img1_np_rz = transform.ToTensor()(img1_PIL_rz).numpy()

    img2_PIL = Image.fromarray(img2)
    img2_PIL_rz = img2_PIL.resize((scaled_im_weight, scaled_im_height))
    img2_np_rz = transform.ToTensor()(img2_PIL_rz).numpy()

    if img1.shape[-1] == 3:
        img1_swap = np.swapaxes(img1, 0, 2)
        img1_swap = np.swapaxes(img1_swap, 1, 2)
    if img2.shape[-1] == 3:
        img2_swap = np.swapaxes(img2, 0, 2)
        img2_swap = np.swapaxes(img2_swap, 1, 2)

    img1_tens = Variable(torch.from_numpy(img1_np_rz).float()).unsqueeze(0)
    img1_tens_nmlz = dlk.normalize_img_batch(img1_tens)
    img2_tens = Variable(torch.from_numpy(img2_np_rz).float()).unsqueeze(0)
    img2_tens_nmlz = dlk.normalize_img_batch(img2_tens)

    dlk_net = dlk.DeepLK(dlk.custom_net(model_path))
    p_lk, _, itr_dlk = dlk_net(img1_tens_nmlz, img2_tens_nmlz, tol=1e-4, max_itr=max_itr_dlk, conv_flag=1, ret_itr=True)

    # 疑问：这里算出来的p_lk是不是能与SIFT算出来的P-lk等价
    p_lk_np = p_lk.cpu().squeeze().detach().numpy()
    p_lk_np = np.append(p_lk_np, 0)
    H_dlk_inv = np.reshape(p_lk_np, (3, 3)) + np.eye(3)
    H_dlk = np.linalg.inv(H_dlk_inv)
    Perspective_img = cv2.warpPerspective(img1, H_dlk, (img1.shape[1], img1.shape[0]))

    img1_rz = img1
    img1_rz_tens = transform.ToTensor()(img1_rz)
    img1_rz_curr_tens = img1_rz_tens.float().unsqueeze(0)
    M_tmpl_w, _, xy_cor_curr_opt = dlk.warp_hmg(img1_rz_curr_tens, p_lk)
    M_tmpl_w_np = M_tmpl_w[0, :, :, :].cpu().detach().numpy()
    temp = np.swapaxes(M_tmpl_w_np, 0, 2)
    M_tmpl_w_np = np.swapaxes(temp, 0, 1)

    return H_dlk