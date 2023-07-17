import numpy as np
import imageio
import torch
from tqdm import tqdm
import sys

import scipy
import scipy.io
import scipy.misc


sys.path.append('Matcher/d2net')
from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale

from PIL import Image

# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

max_edge = 0
max_sum_edges = 2800
preprocessing = 'caffe'
multiscale = True
model_file = './Matcher/d2net/models/d2_tf.pth'
use_relu = True


def d2net_extractor(img_path):
    model = D2Net(
        model_file=model_file,
        use_relu=use_relu,
        use_cuda=use_cuda
    )

    image = imageio.imread(img_path)
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)
    resized_image = image
    if max(resized_image.shape) > max_edge:
        # resize anyway
        set_max_edge = 500
        resized_image_pil = Image.fromarray(resized_image)
        resized_image_size = tuple((np.array(resized_image_pil.size)
                                    * set_max_edge / max(resized_image.shape)).astype(int))
        resized_image = np.array(resized_image_pil.resize(resized_image_size, Image.BILINEAR))

    if sum(resized_image.shape[: 2]) > max_sum_edges:
        resized_image_pil = Image.fromarray(resized_image)
        resized_image_size = tuple((np.array(resized_image_pil.size)
                                    * max_sum_edges / sum(resized_image.shape[: 2])).astype(int))
        resized_image = np.array(resized_image_pil.resize(resized_image_size, Image.BILINEAR))
        # resized_image = scipy.misc.imresize(
        #     resized_image,
        #     max_sum_edges / sum(resized_image.shape[: 2])
        # ).astype('float')

    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]

    input_image = preprocess_image(
        resized_image,
        preprocessing=preprocessing
    )
    with torch.no_grad():
        if multiscale:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model
            )
        else:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model,
                scales=[1]
            )

    # Input image coordinates
    # To save the reshape, set the fact to 1
    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j
    # i, j -> u, v
    keypoints = keypoints[:, [1, 0, 2]]

    return [keypoints, scores, descriptors]


if __name__ == "__main__":
    pass
