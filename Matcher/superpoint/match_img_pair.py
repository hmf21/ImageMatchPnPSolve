import argparse
import glob
import numpy as np
import os
import time

import cv2
import torch
from matplotlib import pyplot as plt

from demo_superpoint import SuperPointFrontend
from demo_superpoint import SuperPointNet


myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])


def match_image_pair():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
    parser.add_argument('input', type=str, default='',
        help='Image directory or movie file or "camera" (for webcam).')
    parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth',
        help='Path to pretrained weights file (default: superpoint_v1.pth).')
    parser.add_argument('--img_glob', type=str, default='*.png',
        help='Glob match if directory of images is specified (default: \'*.png\').')
    parser.add_argument('--skip', type=int, default=1,
        help='Images to skip if input is movie or directory (default: 1).')
    parser.add_argument('--show_extra', action='store_true',
        help='Show extra debug outputs (default: False).')
    parser.add_argument('--H', type=int, default=120,
        help='Input image height (default: 120).')
    parser.add_argument('--W', type=int, default=160,
        help='Input image width (default:160).')
    parser.add_argument('--display_scale', type=int, default=2,
        help='Factor to scale output visualization (default: 2).')
    parser.add_argument('--min_length', type=int, default=2,
        help='Minimum length of point tracks (default: 2).')
    parser.add_argument('--max_length', type=int, default=5,
        help='Maximum length of point tracks (default: 5).')
    parser.add_argument('--nms_dist', type=int, default=4,
        help='Non Maximum Suppression (NMS) distance (default: 4).')
    parser.add_argument('--conf_thresh', type=float, default=0.015,
        help='Detector confidence threshold (default: 0.015).')
    parser.add_argument('--nn_thresh', type=float, default=0.7,
        help='Descriptor matching threshold (default: 0.7).')
    parser.add_argument('--camid', type=int, default=0,
        help='OpenCV webcam video capture ID, usually 0 or 1 (default: 0).')
    parser.add_argument('--waitkey', type=int, default=1,
        help='OpenCV waitkey time in ms (default: 1).')
    parser.add_argument('--cuda', action='store_true',
        help='Use cuda GPU to speed up network processing speed (default: False)')
    parser.add_argument('--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely (default: False).')
    parser.add_argument('--write', action='store_true',
        help='Save output frames to a directory (default: False)')
    parser.add_argument('--write_dir', type=str, default='tracker_outputs/',
        help='Directory where to write output frames (default: tracker_outputs/).')
    opt = parser.parse_args()
    print(opt)

    # This class helps load input images from different sources.
    print('==> Loading pre-trained network.')
    # This class runs the SuperPoint network and processes its outputs.
    fe = SuperPointFrontend(weights_path=opt.weights_path,
                          nms_dist=opt.nms_dist,
                          conf_thresh=opt.conf_thresh,
                          nn_thresh=opt.nn_thresh,
                          cuda=opt.cuda)
    print('==> Successfully loaded pre-trained network.')

    img1 = cv2.imread('data/match_patch/1-map.jpg', 0)
    img1_color = cv2.imread('data/match_patch/1-map.jpg')
    img2 = cv2.imread('data/match_patch/1-drone.jpg', 0)
    img2_color = cv2.imread('data/match_patch/1-drone.jpg')
    interp = cv2.INTER_AREA
    img1 = cv2.resize(img1, (img1.shape[1], img1.shape[0]), interpolation=interp)
    img1 = (img1.astype('float32') / 255.)
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=interp)
    img2 = (img2.astype('float32') / 255.)

    start1 = time.time()
    pts1, desc1, heatmap1 = fe.run(img1)
    end1 = time.time()
    print('==> Processing Time ', format(end1 - start1))
    pts2, desc2, heatmap2 = fe.run(img2)

    # Use Superpoint match
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    desc1 = np.swapaxes(desc1, 0, 1)
    desc2 = np.swapaxes(desc2, 0, 1)
    matches = flann.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.95 * n.distance:
            good.append(m)

    image_1_kpts = []
    pts1 = np.swapaxes(pts1, 0, 1)
    for keypoint in pts1:
        image_1_kpts.append(cv2.KeyPoint(x=keypoint[0], y=keypoint[1], size=1))
    image_2_kpts = []
    pts2 = np.swapaxes(pts2, 0, 1)
    for keypoint in pts2:
        image_2_kpts.append(cv2.KeyPoint(x=keypoint[0], y=keypoint[1], size=1))

    src_pts = np.float32([image_1_kpts[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([image_2_kpts[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H_found, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    gray_kp = cv2.drawKeypoints(img1_color, image_1_kpts, None)
    cv2.imwrite('data/match_patch/1_map_keypoints.jpg', gray_kp)
    gray_dst_kp = cv2.drawKeypoints(img2_color, image_2_kpts, None)
    cv2.imwrite('data/match_patch/1_drone_keypoints.jpg', gray_dst_kp)

    plt.subplot(121), plt.imshow(gray_kp), plt.title('Input')
    plt.subplot(122), plt.imshow(gray_dst_kp), plt.title('Output')
    plt.show()

    # 这里只有用bf这种匹配子貌似才能画出正确的匹配结果
    # image_1_kpts = kp1
    # image_2_kpts = kp2
    # image_1_descriptors = des1
    # image_2_descriptors = des2

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    match_res = cv2.drawMatches((img1_color).astype('uint8'), image_1_kpts,
                                (img2_color).astype('uint8'), image_2_kpts,
                                matches[:30], None, flags=2)
    plt.plot(), plt.imshow(match_res)
    plt.show()
    cv2.imwrite('data/match_patch/1_match_result.jpg', match_res)

    Perspective_img = cv2.warpPerspective(img1_color, H_found, (img1_color.shape[1], img1_color.shape[0]))
    plt.imshow(Perspective_img)
    plt.show()
    cv2.imwrite('data/match_patch/1_warped.jpg', Perspective_img)


    print('==> Finshed Demo.')


if __name__ == '__main__':
    # 证明是可以使用Superpoint进行特征描述的
    match_image_pair()
