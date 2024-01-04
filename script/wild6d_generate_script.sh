#!/bin/bash
# use "bash" rather than "sh" to run this script to avoid syntax error

base_dir="/data4/cxx/dataset/Wild6D/test_set/"
save_dir="/data4/cxx/dataset/Wild6D_manage/test/"
color_from=".jpg"
color_to="_color.png"
depth_from="-depth.png"
depth_to="_depth.png"
mask_from="-mask.png"
mask_to="_oldmask.png"

# 定义第一层目录
first_level_dirs=("bottle" "bowl" "camera" "laptop" "mug")

# 遍历每个第一层目录
for first_dir in "${first_level_dirs[@]}"; do
    # 初始化第一层目录的计数器
    first_level_count=0

    # 查找并排序第二层目录
    second_level_dirs=$(find "$base_dir$first_dir" -mindepth 1 -maxdepth 1 -type d | sort -V)
    for second_dir in $second_level_dirs; do
        # 查找并排序第三层目录
        third_level_dirs=$(find "$second_dir" -mindepth 1 -maxdepth 1 -type d | sort -V)
        for third_dir in $third_level_dirs; do
            # 查找第四层的 'images' 目录
            if [ -d "$third_dir/images" ]; then
                echo "$third_dir/images"
                # 对第一层目录下的处理次数进行计数
                first_level_count=$((first_level_count + 1))

                color_cmd="script/wild6d_relabel_color.sh $third_dir/images/ $save_dir $color_from $color_to"
                depth_cmd="script/data_relabel_suffix.sh $third_dir/images/ $save_dir $depth_from $depth_to"
                mask_cmd="script/data_relabel_suffix.sh $third_dir/images/ $save_dir $mask_from $mask_to"

                echo $color_cmd
                echo $depth_cmd
                echo $mask_cmd

                bash $color_cmd
                bash $depth_cmd
                bash $mask_cmd
            fi
        done
    done

    # 打印第一层目录下处理的目录数量
    echo "Processed $first_level_count directories in $first_dir"
done
