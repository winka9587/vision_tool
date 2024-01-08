#!/bin/bash
# 为每个序列的每个文件生成单独的pkl文件

conda activate SGAP_pack

base_dir="/data4/cxx/dataset/Wild6D/test_set/pkl_annotations/"
save_folder="/data4/cxx/dataset/Wild6D_manage/test/"  # 指定保存文件夹的路径
dataset_root_path="/data4/cxx/dataset/Wild6D/test_set/"

# 文件夹名称和对应代码的字典
declare -A folder_codes=(
    ["bottle"]="01"
    ["bowl"]="02"
    ["camera"]="03"
    ["can"]="04"
    ["laptop"]="05"
    ["mug"]="06"
)

# 遍历每个子文件夹
for category in "${!folder_codes[@]}"; do
    category_code=${folder_codes[$category]}
    category_dir="${base_dir}${category}/"

    # 检查子文件夹是否存在
    if [ -d "$category_dir" ]; then
        # 遍历该子文件夹下的所有 .pkl 文件
        for file in "$category_dir"*.pkl; do
            # 提取文件名，去掉路径和扩展名
            filename=$(basename -- "$file" .pkl)
            
            # 根据横线分割文件名
            IFS='-' read -r -a parts <<< "$filename"

            # 构建新的文件名部分
            new_filename="${category_code}${parts[1]}"
            if [[ ${parts[2]} =~ ^[0-9]{4} ]]; then
                # 处理日期格式的部分
                for (( i = 2; i < ${#parts[@]}; i++ )); do
                    new_filename+=$(echo "${parts[i]}" | tr -d '-')
                done
            else
                # 处理普通数字部分
                new_filename+=$(printf "%02d" "${parts[2]}")
            fi

            # 构建最终的保存路径（不包括 .pkl）
            final_save_path="${save_folder}/scene_${new_filename}"
            
            # 处理文件到新路径
            if [ -d "$final_save_path" ]; then
                echo "Dir already exists: $final_save_path"
                echo "Load $file file to sequence $final_save_path"
                python dataset_utils/wild6d_pose_convert.py "$file" "$final_save_path" "$dataset_root_path"
            else
                echo "!!!!!!!!!!! Dir Not exists: $final_save_path"
            fi
        done
    fi
done
