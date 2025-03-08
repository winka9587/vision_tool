import os
import sys

# 设置默认编码为UTF-8
if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding('utf-8')

# 指定根目录
root_dir = "/data4/cxx/dataset/Wild6D_manage/test/"

# 存储没有_mask.png文件的子目录名
missing_mask_dirs = []

# 遍历根目录下的所有子目录
for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    if os.path.isdir(subdir_path):
        # 检查子目录中是否有_mask.png结尾的文件
        mask_files = [f for f in os.listdir(subdir_path) if f.endswith('_mask.png')]
        if not mask_files:
            # 如果没有，则将子目录名添加到列表中
            missing_mask_dirs.append(subdir)

# 将结果保存到文本文件中
output_file = "missing_mask_dirs.txt"
with open(output_file, 'w', encoding='utf-8') as file:
    for dir_name in missing_mask_dirs:
        file.write(dir_name + '\n')

print("save mask missing folder name to {}".format(output_file))
