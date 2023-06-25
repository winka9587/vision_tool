#!/bin/bash

# 接收命令行参数作为图片文件夹路径
image_folder=$1

# 设置计数器初始值
counter=1

# 检查是否提供了图片文件夹路径参数
if [ -z "$image_folder" ]; then
    echo "Please provide the image folder path as an argument."
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

# 遍历排序后的文件列表，并重命名文件
for file in "${sorted_files[@]}"; do
    if [ -f "$file" ]; then
        # 生成新的文件名，使用四位数序号，例如第1张图像为0001.png
        new_name=$(printf "%04d.png" $counter)
        # 重命名文件
        mv "$file" "$new_name"
        echo "Renamed $file to $new_name"
        # 计数器递增
        counter=$(expr $counter + 1)
    fi
done
