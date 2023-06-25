#!/bin/bash
# 接收命令行参数作为目标目录路径
target_dir=$1
output_file=$2
# 检查是否提供了两个参数
if [ -z "$1" ] || [ -z "$2" ]; then
    echo $1
    echo $2
    echo "Please provide both parameter 1(target dir) and parameter 2(save file name)."
    exit 1
fi

# 进入目标目录
cd "$target_dir"

# 创建保存文件名的txt文件
# output_file="image_names.txt"
> "$output_file"  # 清空文件内容

# 查找目标目录下的所有PNG图像文件，并将文件名保存到txt文件中
# find . -type f -name "*.png" -exec basename {} \; >> "$output_file"
# 排序后存入
ls -1v *.png >> "$output_file"

echo "PNG image names have been saved to $output_file."
