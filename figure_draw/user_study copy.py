import matplotlib.pyplot as plt

# 每一组有多个自定义属性
# 每个属性有均值和方差两个值
# 每个属性的均值和方差用柱状图和误差线表示
default_group_setting = {
    'mean_column_colors': [
        # [i/255.0 for i in [156, 211, 232]],
        # [i/255.0 for i in [243, 147, 147]],
        # [i/255.0 for i in [251, 216, 137]],
        # [i/255.0 for i in [177, 219, 158]]
        [i/255.0 for i in [126, 153, 244]],
        [i/255.0 for i in [204, 124, 113]],
        [i/255.0 for i in [122, 182, 83]],
        [i/255.0 for i in [219, 180, 40]]
    ],  # 默认的颜色设置
    'EV_visible': [[True, True], [True, True], [True, True], [True, True]],  # 默认的均值和方差是否可见
    # 'error_line_color': [i/255.0 for i in [56, 83, 170]],
    'error_line_color': [i/255.0 for i in [189, 72, 53]],
    'mean_font_size': 12,  # 修改均值字体大小
    'error_font_size': 12,  # 修改方差字体大小
    'legend_fontsize': 16, 
    'mean_font_color': [0, 0, 0],  # 修改均值字体颜色
    'error_font_color': [0.9, 0, 0.2]  # 修改方差字体颜色
}

class Group:
    def __init__(self, group_name, setting):
        self.group_name = group_name
        self.setting = setting
        self.data = {}
    
    def update(self, key, value):
        self.data[key] = value
        
    def get(self, key):
        return self.data[key]
    
    def get_keys(self):
        return self.data.keys()

# Enjoyment_E = [2.00, 4.17, 3.33, 3.00, 2.00]
# Enjoyment_V = [0.82, 0.69, 0.47, 0.81, 0.82]
# Focus_E = [3.00, 4.17, 3.17, 3.33, 2.50]
# Focus_V = [0.82, 0.89, 0.90, 0.47, 0.50]
# Friendliness_E = [1.33, 4.67, 3.00, 3.17, 2.66]
# Friendliness_V = [0.94, 0.47, 0.58, 0.69, 1.11]
# Usability_E = [3.00, 4.17, 4.00, 3.50, 3.00]
# Usability_V = [0.41, 1.21, 0.81, 0.76, 0.58]
Enjoyment_E = [1.50, 3.33, 3.00, 2.00, 4.17]
Enjoyment_Min = [1.00, 3.00, 2.00, 1.00, 3.00]
Enjoyment_Max = [3.00, 4.00, 4.00, 3.00, 5.00]
Enjoyment_V = [0.76, 0.47, 0.82, 0.82, 0.90]

Focus_E = [3.00, 3.17, 3.33, 2.50, 4.17]
Focus_Min = [2.00, 2.00, 3.00, 2.00, 2.00]
Focus_Max = [4.00, 4.00, 4.00, 3.00, 5.00]
Focus_V = [0.58, 0.90, 0.47, 0.50, 1.21]

Friendliness_E = [1.17, 3.00, 3.16, 2.67, 4.67]
Friendliness_Min = [1.00, 2.00, 2.00, 2.00, 4.00]
Friendliness_Max = [2.00, 4.00, 4.00, 5.00, 5.00]
Friendliness_V = [0.37, 0.58, 0.69, 1.11, 0.47]

Usability_E = [3.17, 4.00, 3.50, 3.00, 4.17]
Usability_Min = [2.00, 3.00, 3.00, 2.00, 3.00]
Usability_Max = [4.00, 5.00, 5.00, 4.00, 5.00]
Usability_V = [0.69, 0.82, 0.76, 0.58, 0.69]

group_names = ["Blueprint", "2D Marker", "Model Fixed", "Manual Switch", "AR Guidance(Ours)"]
all_attributes = ['Enjoyment', 'Focus', 'Friendliness', 'Usability']
all_attributes_values = {}

all_attributes_values['Enjoyment'] = [Enjoyment_E, Enjoyment_V]
all_attributes_values['Focus'] = [Focus_E, Focus_V]
all_attributes_values['Friendliness'] = [Friendliness_E, Friendliness_V]
all_attributes_values['Usability'] = [Usability_E, Usability_V]

group_num = 5
groups = []
for i in range(group_num):
    g = Group('Group ' + str(i), default_group_setting)
    for attribute in all_attributes:
        g.update(attribute+"_E", all_attributes_values[attribute][0][i])
        g.update(attribute+"_V", all_attributes_values[attribute][1][i])
    groups.append(g)

# width, height
plot_width = 16.5  # default 10 16.5
plot_height = 6.5  # default 10
# fig, ax = plt.subplots(figsize=(plot_width, plot_height))
# plt.subplots_adjust(left=0.04, right=0.98, top=0.98, bottom=0.05)
# x = []
# group_centers = []
# bar_width = 0.25
# inner_gap = 0.05 # 0.08
# group_gap = 0.5
# group_width = len(all_attributes) * (bar_width + inner_gap) + group_gap

# for i in range(group_num):
#     start = i * group_width
#     for j in range(len(all_attributes)):
#         x.append(start + j * (bar_width + inner_gap))
#     group_centers.append(start + (len(all_attributes) * (bar_width + inner_gap) - inner_gap) / 2)

# means = []
# errors = []
# color_list = []

# for i, group in enumerate(groups):
#     means += [group.get(attribute+"_E") for attribute in all_attributes]
#     errors += [group.get(attribute+"_V") for attribute in all_attributes]
#     color_list += group.setting['mean_column_colors']

# ax.bar(x, means, width=bar_width, color=color_list, edgecolor='none')
# ax.set_xticks(group_centers)
# # ax.set_xticklabels([f"Group {i+1}" for i in range(group_num)], rotation=0, ha="center")
# ax.set_xticklabels(group_names, rotation=0, ha="center", fontweight='bold', fontfamily='Times New Roman')

# # 设置x轴字体大小
# ax.tick_params(axis='x', labelsize=16)
# ax.tick_params(axis='y', labelsize=16)
# ax.set_ylabel('Score', fontsize=16, fontfamily='Times New Roman', fontweight='bold') # 修改
# ax.tick_params(axis='y', labelsize=default_group_setting['mean_font_size'], labelcolor=default_group_setting['mean_font_color'])
# ax.errorbar(x, means, yerr=errors, fmt='k,', ecolor=default_group_setting['error_line_color'], elinewidth=1, capsize=5)

# ax.set_ylim(-0.15, 5.9)  # Set the y-axis limits from 0 to 5

# # 假设这部分代码在完整代码的结尾部分，添加均值和方差标签
# for i, (mean, error) in enumerate(zip(means, errors)):
#     error_top = mean + error - 0.1  # 计算误差线的顶端位置
#     ax.text(x[i], error_top + 0.25, f"{mean:.2f}", ha='center', va='bottom', 
#                 fontsize=default_group_setting['mean_font_size'], color=tuple(default_group_setting['mean_font_color']))
#     # 添加方差的标签
#     ax.text(x[i], error_top + 0.1, f"{error:.2f}", ha='center', va='bottom', 
#             fontsize=default_group_setting['error_font_size'], color=tuple(default_group_setting['error_font_color']))

# # # Define color labels
# # color_labels = ['Enjoyment', 'Focus', 'Friendliness', 'Usability']
# # # Define the colors corresponding to each label
# # color_patches = [plt.Line2D([0], [0], color=color, lw=4) for color in default_group_setting['mean_column_colors']]
# # # add legend
# # ax.legend(color_patches, color_labels, loc='upper left')

# color_labels = ['Enjoyment Mean and S.D.', 'Focus Mean and S.D.', 'Friendliness Mean and S.D.', 'Usability Mean and S.D.']
# xwidth = 0.15
# xgap = 0.02
# _, ax2 = plt.subplots(figsize=(plot_width, plot_height))

# # ax2 = ax.twinx()  # create second y-axis, avoid add new errorbar to origin fig
# legend_elements = [(ax2.bar(i - xwidth - xgap, 0, width=xwidth, label='$p(\Delta x)$', color=default_group_setting['mean_column_colors'][i]), 
#                     ax2.errorbar(i - xwidth - xgap, 0, yerr=0, capsize=2, elinewidth=1, fmt='k,', ecolor=default_group_setting['error_font_color'])) for i in range(len(color_labels))]
# # legend_elements = [(ax.bar(i - xwidth - xgap, 0, width=xwidth, label='$p(\Delta x)$', color=default_group_setting['mean_column_colors'][i])) for i in range(len(color_labels))]
# main_legend = ax.legend(legend_elements, color_labels, prop={'family': 'Times New Roman', 'weight': 'bold', 'size': default_group_setting['legend_fontsize']})
# # main_legend = ax.legend(legend_elements, color_labels, fontsize=30)
# # main_legend.set_bbox_to_anchor((0, 1, 0.3, 0.4))  # 调整图例框的位置和尺寸
# # main_legend.set_loc('upper left')  # 图例框的左中对齐


# # plt.tight_layout()
# plt.show()


fig, ax = plt.subplots(figsize=(plot_width, plot_height))
plt.subplots_adjust(left=0.04, right=0.98, top=0.98, bottom=0.05)
x = []
group_centers = []
bar_width = 0.25
inner_gap = 0.05
group_gap = 0.5
group_width = len(all_attributes) * (bar_width + inner_gap) + group_gap

for i in range(group_num):
    start = i * group_width
    for j in range(len(all_attributes)):
        x.append(start + j * (bar_width + inner_gap))
    group_centers.append(start + (len(all_attributes) * (bar_width + inner_gap) - inner_gap) / 2)

# 绘制两段柱状图
for i, group in enumerate(groups):
    for j, attribute in enumerate(all_attributes):
        base_x = x[i * len(all_attributes) + j]
        E_value = group.get(attribute+"_E")
        Min_value = globals()[attribute + "_Min"][i]
        Max_value = globals()[attribute + "_Max"][i]

        # 第一段: Min 到 E
        ax.bar(base_x, E_value - Min_value, bottom=Min_value, width=bar_width, 
               color=group.setting['mean_column_colors'][j], edgecolor='none')

        # 第二段: E 到 Max
        ax.bar(base_x, Max_value - E_value, bottom=E_value, width=bar_width, 
               color=group.setting['mean_column_colors'][j], alpha=0.5, edgecolor='none')

# 设置x轴标签等
ax.set_xticks(group_centers)
ax.set_xticklabels(group_names, rotation=0, ha="center", fontweight='bold', fontfamily='Times New Roman')

ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.set_ylabel('Score', fontsize=16, fontfamily='Times New Roman', fontweight='bold')
ax.set_ylim(-0.15, 5.9)

# 绘制误差线
for i in range(len(x)):
    ax.errorbar(x[i], means[i], yerr=errors[i], fmt='k,', ecolor=default_group_setting['error_line_color'], elinewidth=1, capsize=5)

# 添加标签
for i, (mean, error) in enumerate(zip(means, errors)):
    error_top = mean + error - 0.1
    ax.text(x[i], error_top + 0.25, f"{mean:.2f}", ha='center', va='bottom', 
                fontsize=default_group_setting['mean_font_size'], color=tuple(default_group_setting['mean_font_color']))
    ax.text(x[i], error_top + 0.1, f"{error:.2f}", ha='center', va='bottom', 
            fontsize=default_group_setting['error_font_size'], color=tuple(default_group_setting['error_font_color']))

plt.show()
