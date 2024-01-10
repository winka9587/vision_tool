# #!/bin/bash

# # 接收命令行参数作为图片文件夹路径
# image_folder=$1
# # 输出路径
# save_folder=$2
# # 指定的原始后缀
# suffix=$3
# # 新的后缀
# new_suffix=$4

# # 设置计数器初始值
# counter=1

# # 检查是否提供了图片文件夹路径参数和所需的参数数量
# if [ -z "$image_folder" ] || [ -z "$save_folder" ] || [ -z "$suffix" ] || [ -z "$new_suffix" ]; then
#     echo "Please provide all four parameters: target dir, save file name, original suffix, and new suffix."
#     exit 1
# fi

# # 进入图片文件夹
# cd "$image_folder"
# echo "image_folder: $image_folder"

# # 获取所有不含 '-depth' 或 '-mask' 的 .png 文件
# # files=($(ls | grep -vE "(-depth|-mask).png" | grep ".jpg$"))  # 排序错误
# files=($(ls | grep -vE "(-depth|-mask).png" | grep ".jpg$" | sort -n -t . -k 1))


# # 打印找到的所有文件
# echo "Found files:"
# for file in "${files[@]}"; do
#     echo "$file"
# done

# # 自定义排序函数，按照数字的大小进行排序
# sort_files() {
#     printf '%s\n' "${files[@]}" | sort -t'.' -k1,1n -k2,2n
# }

# # 调用自定义排序函数，获取排序后的文件列表
# sorted_files=($(sort_files))

# # 定义文件夹名称和对应的代码的字典
# declare -A folder_codes=(
#     ["bottle"]="01"
#     ["bowl"]="02"
#     ["camera"]="03"
#     ["can"]="04"
#     ["laptop"]="05"
#     ["mug"]="06"
# )

# # 提取路径的相关部分
# # 假设路径格式为 /data4/cxx/dataset/Wild6D/test_set/bottle/0001/1/images/
# sub_path=$(echo "$image_folder" | sed 's|.*/Wild6D/test_set/\(.*\)/images/|\1|')

# # 分解子路径为其组成部分
# IFS='/' read -r -a path_parts <<< "$sub_path"

# # 替换物品名称为对应的数字代码
# folder_code=${folder_codes[${path_parts[0]}]}

# # 格式化子路径的最后部分
# last_part=$(printf "%02d" ${path_parts[2]})

# # 生成新的子路径
# new_sub_path="${folder_code}${path_parts[1]}${last_part}"

# # 生成最终的保存路径
# final_save_folder="$save_folder/scene_$new_sub_path"

# # 检查并创建最终的保存路径
# if [ ! -d "$final_save_folder" ]; then
#     mkdir -p "$final_save_folder"
#     echo "Created directory: $final_save_folder"
# else
#     echo "Directory already exists: $final_save_folder"
# fi

# # 遍历找到的文件列表，并重命名文件
# for file in "${files[@]}"; do
#     if [ -f "$file" ]; then
#         # 生成新的文件名，使用四位数序号，并添加 '_color' 后缀
#         new_name=$(printf "%04d_color.png" $counter)
#         # 复制文件到目标目录
#         cp "$file" "$final_save_folder/$new_name"
#         echo "Renamed $file to $new_name"
#         # 计数器递增
#         counter=$(expr $counter + 1)
#     fi
# done

#!/bin/bash

# 接收命令行参数作为图片文件夹路径
image_folder=$1
# 输出路径
save_folder=$2
# 指定的原始后缀
suffix=$3
# 新的后缀
new_suffix=$4

# 检查是否提供了图片文件夹路径参数和所需的参数数量
if [ -z "$image_folder" ] || [ -z "$save_folder" ] || [ -z "$suffix" ] || [ -z "$new_suffix" ]; then
    echo "Please provide all four parameters: target dir, save file name, original suffix, and new suffix."
    exit 1
fi

# 定义文件夹名称和对应的代码的字典
declare -A folder_codes=(
    ["bottle"]="01"
    ["bowl"]="02"
    ["camera"]="03"
    ["can"]="04"
    ["laptop"]="05"
    ["mug"]="06"
)

# 进入图片文件夹
cd "$image_folder" || exit
echo "image_folder: $image_folder"

# 获取所有不含 '-depth' 或 '-mask' 的 .jpg 文件的基础名，并按数字排序
base_names=()
for file in $(ls | grep -vE "(-depth|-mask).png" | grep ".jpg$" | sort -n -t . -k 1); do
    if [ -f "$file" ]; then
        base_name=$(basename "$file" "$suffix")
        base_names+=("$base_name")
    fi
done

# 对基础名进行数值排序
IFS=$'\n' sorted_base_names=($(sort -n <<<"${base_names[*]}"))
unset IFS

# 提取路径的相关部分
# 假设路径格式为 
# /data4/cxx/dataset/Wild6D/test_set/bottle/0001/1/images/
# 或
# /data4/cxx/dataset/Wild6D/test_set/bottle/0034/2021-09-16--18-45-29/images/ 或 /rgbd/
sub_path=$(echo "$image_folder" | sed 's|.*/Wild6D/test_set/\(.*\)/rgbd/|\1|')
if [[ "$sub_path0" == "$image_folder" ]]; then
    sub_path=$(echo "$image_folder" | sed 's|.*/Wild6D/test_set/\(.*\)/images/|\1|')
fi

# 分解子路径为其组成部分
IFS='/' read -r -a path_parts <<< "$sub_path"

# 替换物品名称为对应的数字代码
folder_code=${folder_codes[${path_parts[0]}]}

# 检查第三层目录名格式，并做相应的转换
# 例如：2021-09-16--18-45-29 转换为 20210916184529
if [[ ${path_parts[2]} =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}-- ]]; then
    path_parts[2]=$(echo "${path_parts[2]}" | tr -d '-' | tr -d ':')
fi

# 格式化子路径的最后部分
last_part=$(printf "%02d" ${path_parts[2]})

# 生成新的子路径
new_sub_path="${folder_code}${path_parts[1]}${last_part}"

# 生成最终的保存路径
final_save_folder="$save_folder/scene_$new_sub_path"

# 创建txt文件存储文件夹之间的映射关系
txt_file="$save_folder/folder_refer.txt"
> "$txt_file"
echo "$image_folder" >> "$txt_file"
echo "$final_save_folder" >> "$txt_file"

# 检查并创建最终的保存路径
if [ ! -d "$final_save_folder" ]; then
    mkdir -p "$final_save_folder"
    echo "Created directory: $final_save_folder"
else
    echo "Directory already exists: $final_save_folder"
fi

# 遍历排序后的基础名列表，并重命名文件
for base_name in "${sorted_base_names[@]}"; do
    original_file="${base_name}${suffix}"
    if [ -f "$original_file" ]; then
        # 生成新的文件名，确保基础名是四位数
        new_name=$(printf "%04d$new_suffix" "$base_name")
        # 复制文件到目标目录
        cp "$original_file" "$final_save_folder/$new_name"
        echo "Copied $original_file to $new_name"
    fi
done
