#!/bin/bash

# 使用示例: bash script/data_relabel.sh /data4/cxx/dataset/gs/storm2/color/ /data4/cxx/dataset/gs/storm2/color2/ 0 5

# 接收命令行参数作为图片文件夹路径
image_folder=$1
# 输出路径
save_folder=$2

# 设置计数器初始值(从0开始还是从1开始计数)
counter=$3  # 0 or 1
fill_len=$4  # 5

# 检查是否提供了图片文件夹路径参数
# 检查是否提供了两个参数
if [ -z "$image_folder" ] || [ -z "$save_folder" ] || [ -z "$counter" ] || [ -z "$fill_len" ]; then
    echo "Please provide all 4 parameters: image_folder, save_folder, counter, fill_len."
    exit 1
fi

# 进入图片文件夹
cd "$image_folder"

# 获取所有图片文件的列表
files=(*.png)

# 自定义排序函数，按照数字的大小进行排序
sort_files() {
    printf '%s\n' "${files[@]}" | sort -t'.' -k1,1n -k2,2n
}

# 调用自定义排序函数，获取排序后的文件列表
sorted_files=($(sort_files))

echo "mkdir to save"
mkdir "$save_folder"

# 遍历排序后的文件列表，并重命名文件
for file in "${sorted_files[@]}"; do
    if [ -f "$file" ]; then
        # 生成新的文件名，使用四位数序号，例如第1张图像为0001.png
        # new_name=$(printf "%04d.png" $counter)
        new_name=$(printf "%0${fill_len}d.png" $counter)
        # 重命名文件
        # 可能会有小bug, 假如有图片原本就是1000, 那么其更新后, 因为起始是1, 变成1001.png, 反而将后面的还未修改的图像覆盖了
        # 解决方案: 输出到另一个目录下

        mv "$file" "$save_folder/$new_name"
        echo "Renamed $file to $new_name"
        # 计数器递增
        counter=$(expr $counter + 1)
    fi
done
