import sys
import os
import numpy as np
import argparse
import cv2

import superpoint.demo_superpoint as superpoint


if __name__ == "__main__":
    PATCHNETVLAD_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    parser = argparse.ArgumentParser(description='Patch-NetVLAD-Feature-Match')
    parser.add_argument('--dataset_file_path', type=str, required=True,
                        help='Full path (with extension) to a text file that stores the save location and name of all images in the dataset folder')
    parser.add_argument('--dataset_root_dir', type=str, default='',
                        help='If the files in dataset_file_path are relative, use dataset_root_dir as prefix.')
    parser.add_argument('--output_features_dir', type=str, default=os.path.join(PATCHNETVLAD_ROOT_DIR, 'output_features'),
                        help='Path to store all patch-netvlad features')
    parser.add_argument('--nocuda', action='store_true', help='If true, use CPU only. Else use GPU.')
    parser.add_argument('--re_scale', type=int, default=None)

    opt = parser.parse_args()
    print(opt)

    interp = cv2.INTER_AREA
    weights_path = './superpoint_v1.pth'
    nms_dist = 4
    conf_thresh = 0.015
    nn_thresh = 0.7
    cuda = True
    fe = superpoint.SuperPointFrontend(weights_path, nms_dist, conf_thresh, nn_thresh, cuda)

    textfile = 'USGS_map_tile_index.txt'
    with open(textfile, 'r') as f:
        image_list = f.read().splitlines()
    output_local_features_prefix = os.path.join(opt.output_features_dir, 'patchfeats')

    for image_name in image_list:
        image = cv2.imread(opt.dataset_root_dir + "\\" + image_name, 0)
        image_color = cv2.imread(opt.dataset_root_dir + "\\" + image_name)
        image = cv2.resize(image, (image.shape[1], image.shape[0]), interpolation=interp)
        image = (image.astype('float32') / 255.)
        keypoints, descriptors, heatmap = fe.run(image)

        filename = PATCHNETVLAD_ROOT_DIR + "\\" + output_local_features_prefix + '_' + (image_name.split('\\'))[-1][:-4] + '.npy'
        with open(filename, 'wb') as output_file:
            np.savez(
                output_file,
                keypoints=keypoints,
                descriptors=descriptors,
                heatmap=heatmap
            )
