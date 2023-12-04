# 海淀机场飞行
# image_dir = "E://Dataset//camera1//"
image_dir = "../haidian_airport/IR_image2/"
image_dir_ext = "*.png"
motion_param_loc = "../haidian_airport/P_airport - 副本.csv"
satellite_map_name = 'haidian_airport'
map_loc = "../haidian_airport/map_airport.tif"
# model_path = "../models/conv_02_17_18_1833.pth"
# model_path = "../deep_feat_VGG16/models/trained_model_output.pth"
model_path = "./deep_feat_VGG16/models/conv_02_17_18_1833.pth"
opt_img_height = 100
img_h_rel_pose = 600
opt_param_save_loc = "../haidian_airport/test_out.mat"
map_resolution = 0.465
# scale_img_to_map = 27.79 / 6
scale_img_to_map = 29.10 / 6

# GPS_1 = [40.0856287594884, 116.10104906676413]
# GPS_2 = [40.0652127940419, 116.116820683836]

GPS_1 = [116.116820683836, 40.0856287594884]
GPS_2 = [116.10104906676413, 40.0652127940419]

device = "cpu"
USE_CUDA = False

# 在定位过程中的一些其他的参数，包括图片提取以及优化过程
tol = 1e-3
max_itr = 10
lam1 = 1
lam2 = 0.01
scaling_for_disp = 1    # orginal 2.25
max_itr_dlk = 150
