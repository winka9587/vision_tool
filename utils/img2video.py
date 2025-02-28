import os.path

import cv2
import glob
import numpy as np

def images_to_video(path):
    img_array = []

    for filename in glob.glob(path + '/*.png'):
        img = cv2.imread(filename)
        if img is None:
            print(filename + " is error!")
            continue
        img_array.append(img)

    #得到文件名
    index = path.rfind('/')
    filename = path[index+1:]
    # 图片的大小需要一致
    size = img_array[0].shape
    fps = 30
    filename = filename + "_" + str(fps) + "fps"
    out = cv2.VideoWriter(filename + ".mp4", cv2.VideoWriter_fourcc('X', '2', '6', '4'), fps, (1280, 512))
    #out = cv2.VideoWriter(filename + ".avi", cv2.VideoWriter_fourcc(*'DIVX'), fps, (640, 512))


    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def arr2video(arr, save_dir, save_name):
    fps = 60
    w = arr[0].shape[1]
    h = arr[0].shape[0]
    save_name = f"{save_dir}/{save_name}_{len(arr)}frames_{fps}fps"
    # 对于小图片,fps不起作用
    out = cv2.VideoWriter(save_name + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    # out = cv2.VideoWriter(save_name + ".mp4", cv2.VideoWriter_fourcc('X', '2', '6', '4'), fps, (w, h))
    print(f'frame {len(arr)}')
    for i in range(len(arr)):
        out.write(arr[i])
    out.release()
    print(f"{save_name}.mp4 has been written, video contains {len(arr)} frames, fps={fps}")


# 添加白边的图像img,添加到top还是bottom,添加的高度
# 返回拼接好的图像
def add_white_to_arr(arr, is_top, add_height):
    res_arr = []
    for img in arr:
        tmp_img_array = []
        h = img.shape[0]
        w = img.shape[1]
        c = img.shape[2]
        h = add_height
        white_img = np.zeros((h, w, c), np.uint8)
        white_img.fill(255)

        if is_top:
            tmp_img_array.append(white_img)
            tmp_img_array.append(img)
        else:
            tmp_img_array.append(img)
            tmp_img_array.append(white_img)
        # 0 沿竖直方向拼接,1沿水平方向拼接
        res_img = np.concatenate(tmp_img_array, 0)
        res_arr.append(res_img)
    return res_arr


# 将两个图像数组拼接,裁剪为最大长度final_length(长度优先选择最短的)
# add_direction = 0:竖直; 1:水平;
def concat_image_and_add_white(arr1, arr2, final_length, is_add_white=False, add_direction=0, white_padding=10):
    final_img_array = []
    video_len = min(len(arr1), len(arr2))
    video_len = min(video_len, video_len)
    print(f"video len{video_len}")
    for i in range(0, video_len):
        img_1 = arr1[i]
        img_2 = arr2[i]
        if add_direction == 0 and img_1.shape[1] != img_2.shape[1]:
            print(f"Error at concat_image_and_add_white, img_1.shape[0]:{img_1.shape} != img_2.shape[0]:{img_2.shape}")
            return
        if add_direction == 1 and img_1.shape[0] != img_2.shape[0]:
            print(f"Error at concat_image_and_add_white, img_1.shape[0]:{img_1.shape} != img_2.shape[0]:{img_2.shape}")
            return
        if is_add_white:
            h = img_1.shape[0]
            w = img_1.shape[1]
            c = img_1.shape[2]
            print(1)
            if add_direction == 0:
                # 水平拼接
                w = white_padding
                white_img = np.zeros((h, w, c), np.uint8)
                white_img.fill(255)
            else:
                h = white_padding
                white_img = np.zeros((h, w, c), np.uint8)
                white_img.fill(255)
            tmp_img_arr = []
            tmp_img_arr.append(img_1)
            tmp_img_arr.append(white_img)
            tmp_img_arr.append(img_2)
            final_img = np.concatenate(tmp_img_arr, add_direction)
            final_img_array.append(final_img)
        else:
            tmp_img_arr = []
            tmp_img_arr.append(img_1)
            tmp_img_arr.append(img_2)
            final_img = np.concatenate(tmp_img_arr, add_direction)
            final_img_array.append(final_img)
    return final_img_array


# 读取path下的图像并返回
# 先裁剪后resize
def read_img(path, need_resize=False,
             w=640, h=512,
             need_cut=False, cut_x_start=0, cut_x_end=0, cut_y_start=0, cut_y_end=0,
             prefix=None,
             suffix=None,
             limit = None):
    read_img_array = []
    if limit:
        # 临时
        i = 0
    for filename in glob.glob(path + '/*.png'):
        if limit:
            if i==limit:
                break
            else:
                i+=1

        if suffix==None:
            pass
        else:
            if not filename.endswith(suffix):
                continue
        if prefix == None:
            pass
        else:
            if not filename.split('/')[-1].split('\\')[-1].startswith(prefix):
                continue
        img = cv2.imread(filename)
        # 裁剪图像
        if need_cut:
            img = img[cut_x_start:cut_x_end, cut_y_start:cut_y_end]
        if need_resize:
            # 注意这里w和h的位置又反过来了
            size = (w, h)
            img = cv2.resize(img, size)
        if img is None:
            print(filename + " is error!")
            continue
        read_img_array.append(img)
        print(f'total read {len(read_img_array)} frame')
    return read_img_array


def test_1():
    print("test test")
    #path_1 = "H:\\DATA-LJC\\MVT Benchmark\\complex_dynamic_suspension\\final\\render_6_Standtube"
    #arr1 = read_img(path_1, True, 320, 256)
    #arr2 = read_img(path_1)0
    #cv2.imshow("test", arr1[0])
    #cv2.imshow("test2", arr2[0])
    #cv2.waitKey(0)


def concat_two_seq():
    path_1 = "H:\\DATA-LJC\\MVT Benchmark\\complex_dynamic_handheld\\final\\render_6_Standtube"
    path_2 = "H:\\DATA-LJC\\MVT Benchmark\\complex_dynamic_suspension\\final\\render_4_Stitch"
    arr1 = read_img(path_1)
    arr2 = read_img(path_2)
    arr3 = concat_image_and_add_white(arr1, arr2, 600, False, 1)
    arr4 = add_white_to_arr(arr3, True, 80)
    arr5 = add_white_to_arr(arr4, False, 80)
    scene_name_1 = path_1.split("\\")[-3]
    obj_name_1 = path_1.split("\\")[-3].split("_")[-1]
    scene_name_2 = path_2.split("\\")[-3]
    obj_name_2 = path_2.split("\\")[-3].split("_")[-1]
    name_1 = scene_name_1 + "-" + obj_name_1
    name_2 = scene_name_2 + "-" + obj_name_2
    arr2video(arr5, "video", f"{name_1}_&&_{name_2}")


def concat_8_seq_indoor():
    path = []
    path.append("H:\\DATA-LJC\\MVT Benchmark\\complex_static_trans\\final\\render_10_Jack")
    path.append("H:\\DATA-LJC\\MVT Benchmark\\easy_static_suspension\\final\\render_14_FlashLight")
    path.append("H:\\DATA-LJC\\MVT Benchmark\\easy_static_handheld\\final\\render_13_Deadpool")
    path.append("H:\\DATA-LJC\\MVT Benchmark\\light_dynamic_handheld\\final\\render_20_Squirrel")
    path.append("H:\\DATA-LJC\\MVT Benchmark\\light_dynamic_suspension\\final\\render_8_Ape")
    path.append("H:\\DATA-LJC\\MVT Benchmark\\light_static_handheld\\final\\render_10_Jack")
    path.append("H:\\DATA-LJC\\MVT Benchmark\\light_static_suspension\\final\\render_3_Vampire queen")
    path.append("I:\\DATA-LJC\\MVT Benchmark\\complex_dynamic_occlusion_2\\final\\render\\13_Deadpool")


    arr = []
    for i in range(len(path)):
        print(i)
        if i== 7:
            arr.append(read_img(path[i], need_resize=True, w=320, h=256, need_cut=True, cut_x_end=640, cut_y_end=512))
        else:
            arr.append(read_img(path[i], True, 320, 256))
        print(len(arr[i]))
    # 将4个为1组拼接起来
    arr_tmp_top = concat_image_and_add_white(arr[0], arr[1], 600, False, 1)
    arr_tmp_top = concat_image_and_add_white(arr_tmp_top, arr[2], 600, False, 1)
    arr_tmp_top = concat_image_and_add_white(arr_tmp_top, arr[3], 600, False, 1)
    arr_tmp_top = add_white_to_arr(arr_tmp_top, True, 80)

    arr_tmp_bottom = concat_image_and_add_white(arr[4], arr[5], 600, False, 1)
    arr_tmp_bottom = concat_image_and_add_white(arr_tmp_bottom, arr[6], 600, False, 1)
    arr_tmp_bottom = concat_image_and_add_white(arr_tmp_bottom, arr[7], 600, False, 1)
    arr_tmp_bottom = add_white_to_arr(arr_tmp_bottom, False, 80)

    arr_final = concat_image_and_add_white(arr_tmp_top, arr_tmp_bottom, 600, False, 0)
    arr2video(arr_final, "video", "8_seq")


def concat_4_seq_outdoor():
    path = []
    path.append("H:\\DATA-LJC\\MVT Benchmark\\outdoor_dynamic_scene1_handheld\\final\\render\\3_Vampire queen")
    path.append("H:\\DATA-LJC\\MVT Benchmark\\outdoor_dynamic_scene1_suspension\\final\\render\\16_Driller")
    path.append("H:\\DATA-LJC\\MVT Benchmark\\outdoor_dynamic_scene2_handheld\\final\\render\\20_Squirrel")
    path.append("H:\\DATA-LJC\\MVT Benchmark\\outdoor_dynamic_scene2_suspension\\final\\render\\10_Jack")
    arr = []
    for i in range(len(path)):
        print(i)
        arr.append(read_img(path[i], True, 640, 256))
        print(len(arr[i]))
    # 将4个为1组拼接起来
    arr_tmp_top = concat_image_and_add_white(arr[0], arr[1], 600, False, 1)
    arr_tmp_top = add_white_to_arr(arr_tmp_top, True, 80)

    arr_tmp_bottom = concat_image_and_add_white(arr[2], arr[3], 600, False, 1)
    arr_tmp_bottom = add_white_to_arr(arr_tmp_bottom, False, 80)

    arr_final = concat_image_and_add_white(arr_tmp_top, arr_tmp_bottom, 600, False, 0)
    arr2video(arr_final, "video", "4_seq_outdoor")


def generate_single_video():
    path = []
    # path.append("H:\\DATA-LJC\\MVT Benchmark\\complex_static_trans\\final\\render_10_Jack")
    # path.append("H:\\DATA-LJC\\MVT Benchmark\\easy_static_suspension\\final\\render_14_FlashLight")
    # path.append("H:\\DATA-LJC\\MVT Benchmark\\easy_static_handheld\\final\\render_13_Deadpool")
    # path.append("H:\\DATA-LJC\\MVT Benchmark\\light_dynamic_handheld\\final\\render_20_Squirrel")
    # path.append("H:\\DATA-LJC\\MVT Benchmark\\light_dynamic_suspension\\final\\render_8_Ape")
    # path.append("H:\\DATA-LJC\\MVT Benchmark\\light_static_handheld\\final\\render_10_Jack")
    # path.append("H:\\DATA-LJC\\MVT Benchmark\\light_static_suspension\\final\\render_3_Vampire queen")
    path.append("I:\\DATA-LJC\\MVT Benchmark\\complex_dynamic_occlusion_2\\final\\render\\13_Deadpool")
    for i in range(len(path)):
        arr1 = read_img(path[i], need_cut=True, cut_x_end=512, cut_y_end=640)
        # arr1 = read_img(path[i])
        scene_name_1 = path[i].split("\\")[-3]
        obj_name_1 = path[i].split("\\")[-1].split("_")[-1]
        name_1 = scene_name_1 + "-" + obj_name_1
        arr2video(arr1, "page_video", f"{name_1}")
    print(f"total {len(path)} videos")


# 输出的video与脚本在同一目录下
def simple_imgs2video(input_img_path, output_dir, output_video_name, arg_prefix=None, arg_suffix=None):
    arr = read_img(input_img_path, prefix=arg_prefix, suffix = arg_suffix)
    arr2video(arr, output_dir, output_video_name)


# 读取概率图绘制在原图上
def simple_imgs2video_prob(rawImage_path, prob_path, output_dir, output_video_name, arg_prefix=None, arg_suffix=None, arg_w=None, arg_h=None):
    arg_limit = None
    raw = read_img(rawImage_path, suffix=arg_suffix, limit=arg_limit)
    color_bar = cv2.imread('M:/TVCG Track/colorbar3.png')
    color_bar_offset = 20
    res = []
    joints = [1, 2, 3, 4, 5]
    imgs = []
    for j in joints:
        path = os.path.join(prob_path, 'ProResultIMG', str(j), "prob")
        arr = read_img(path, prefix=arg_prefix, suffix=arg_suffix, limit=arg_limit)
        # assert len(arr) == len(raw)
        imgs.append(arr)
    # 拼接图片
    # (x,y)
    base = (300, 200)
    img_pos = [(base[0]+60, base[1]+50),
               (base[0]+60, base[1]+300),
               (base[0]+300, base[1]+90),
               (base[0]+300, base[1]+350),
               (base[0]+300, base[1]+500)]
    for raw_i in range(len(raw)):
        # 限制
        if raw_i == 580:
            break
        raw_img = raw[raw_i]
        # y, x
        bg = raw_img[base[0]-200:base[0] + 500, base[1]:base[1] + 550, :]
        bg2 = raw_img[base[0] - 200:base[0] + 500, base[1]:base[1] + 550, :].copy()
        bg[:, :, 0].fill(220)
        bg[:, :, 1].fill(220)
        bg[:, :, 2].fill(220)
        bg[:, :, 0] = bg[:, :, 0] + bg2[:, :, 0]/2
        bg[:, :, 1] = bg[:, :, 1] + bg2[:, :, 1]/2
        bg[:, :, 2] = bg[:, :, 2] + bg2[:, :, 2]/2
        bg[bg < 220] = 220
        bg[bg.shape[0]-185-color_bar_offset:bg.shape[0]-color_bar_offset, bg.shape[1]-28-color_bar_offset:bg.shape[1]-color_bar_offset, :] = color_bar
        for i in range(len(img_pos)):
            # 背景
            p = img_pos[i]
            prob_ = imgs[i][raw_i]

            half_x_1 = prob_.shape[1] - int(prob_.shape[1]/2)
            half_x_2 = int(prob_.shape[1]/2)
            half_y_1 = prob_.shape[0] - int(prob_.shape[0]/2)
            half_y_2 = int(prob_.shape[0]/2)


            x_1 = max(0, p[0]-half_x_1)
            x_2 = min(raw_img.shape[1], p[0]+half_x_2)
            y_1 = max(0, p[1]-half_y_1)
            y_2 = min(raw_img.shape[0], p[1]+half_y_2)
            raw_img[y_1:y_2, x_1:x_2, :] = prob_[0:y_2-y_1, 0:x_2-x_1, :]
        # cv2.imshow("1", raw_img)
        # cv2.waitKey(0)
        res.append(raw_img)
    arr2video(res, output_dir, output_video_name)


def concate_img_with_white(imgs, add_direction, white_padding=10):
    img_1 = imgs[0]
    h = img_1.shape[0]
    w = img_1.shape[1]
    c = img_1.shape[2]
    print(1)
    if add_direction == 1:
        # 1 水平拼接
        w = white_padding
        white_img = np.zeros((h, w, c), np.uint8)
        white_img.fill(255)
    else:
        # 0 竖直
        h = white_padding
        white_img = np.zeros((h, w, c), np.uint8)
        white_img.fill(255)

    tmp_img_arr = []
    for i in range(len(imgs)-1):
        tmp_img_arr.append(imgs[i])
        tmp_img_arr.append(white_img)
    tmp_img_arr.append(imgs[-1])
    final_img = np.concatenate(tmp_img_arr, add_direction)

    return final_img

# 将图片拼接
def simple_read_and_concate_imgs():
    # 1.
    # path = "D:/2022/WB_paper_data/track_gt_compare/final/"
    # folder_name = "cat"
    # direction = 1
    # 2.
    # path = "D:/2022/WB_paper_data/track_gt_compare/final/"
    # folder_name = "lego"
    # direction = 1
    # 3.
    # path = "D:/2022/WB_paper_data/track_gt_compare/final/"
    # folder_name = "squirrel"
    # direction = 1
    # 4.
    # path = "D:/2022/WB_paper_data/track_gt_compare/"
    # folder_name = "version3"
    # direction = 0

    # line1
    path = "D:/2022/WB_paper_data//"
    folder_name = "grasp_2x6"
    direction = 0

    imgs = []
    for filename in glob.glob(os.path.join(path, folder_name) + '/*.png'):
        img = cv2.imread(filename)
        imgs.append(img)
        print(f"read img {filename}")
    res = concate_img_with_white(imgs, direction)
    cv2.imwrite(os.path.join(path, folder_name + ".png"), res)


def main():
    # concat_two_seq()
    # concat_8_seq_indoor()
    # concat_4_seq_outdoor()
    # generate_single_video()

    # simple_imgs2video("E:\\Image\\drawArm\\cat5_gt", "E:\\Image\\drawArm", "cat5_gt")
    simple_imgs2video("F:\\2024_11_06_13_50_251730872230.678803\\color", "F:\\2024_11_06_13_50_251730872230.678803\\", "wheel")

    # simple_imgs2video("F:/NOCS_DataSet/real_train/real_train/scene_1", "CAPTRA", "real_train_scene_1", '_color.png')

    # 基础生成视频功能
    # simple_imgs2video("E:/Onedrive/obj_pcd/Seq_viz_cur/Real/train/can", "pcd", "can_tall_yellow_norm", arg_suffix='.png')

    # 关节概率图生成视频
    # perfix = 'prob_'
    # simple_imgs2video_prob("M:/TVCG Track/aubo/squirrel/test_1/track_gt/",
    #                        "M:/TVCG Track/squirrelTest1/",
    #                        "wb_video",
    #                         'final_prob_squ_test1_519', arg_prefix=perfix, arg_suffix='.png')

    # 拼接图片
    # simple_read_and_concate_imgs()


if __name__ == "__main__":
    main()



