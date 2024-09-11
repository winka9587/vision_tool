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
        [i/255.0 for i in [219, 180, 40]],
        [i/255.0 for i in [251, 216, 137]], # new added color
        [i/255.0 for i in [177, 219, 158]]
    ],  # 默认的颜色设置
    # 'EV_visible': [[True, True], [True, True], [True, True], [True, True]],  # 默认的均值和方差是否可见
    'EV_visible': [[True, True], [True, True], [True, True], [True, True], [True, True], [True, True]],  # 默认的均值和方差是否可见
    # 'error_line_color': [i/255.0 for i in [56, 83, 170]],
    'error_line_color': [i/255.0 for i in [189, 72, 53]],
    'mean_font_size': 14,  # 修改均值字体大小
    'error_font_size': 14,  # 修改方差字体大小
    'legend_fontsize': 18, 
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

# 
# Efficiency 665.00 (76.89) 313.50 (48.34) 457.58 (30.10) 851.80 (89.48) 376.50 (58.19)
# Effectiveness 0.80 (0.16) 0.97 (0.074) 0.93 (0.074) 0.58 (0.21) 0.96 (0.038)
# Error Count 2.33 (0.47) 0.50 (0.76) 1.00 (0.58) 0.83 (0.37) 2.16 (1.06)
# Judgment Time 381.33 (58.63) 17.00 (3.96) 20.23 (4.58) 22.16 (5.01) 20.00 (3.74)
# Assembly Time 313.67 (61.88) 296.50 (48.63) 437.35 (29.10) 829.67 (87.55) 356.58 (56.34)
# Tracking Failure — 0.17 (0.37) — 3.83 (1.34) 0.67 (0.74)
# Efficiency_E = [665.00, 313.50, 457.58, 851.80, 376.50]
# Efficiency_V = [76.89, 48.34, 30.10, 89.48, 58.19]
# Effectiveness_E = [0.80, 0.97, 0.93, 0.58, 0.96]
# Effectiveness_V = [0.16, 0.074, 0.074, 0.21, 0.038]
# Error_Count_E = [2.33, 0.50, 1.00, 0.83, 2.16]
# Error_Count_V = [0.47, 0.76, 0.58, 0.37, 1.06]
# Judgment_Time_E = [381.33, 17.00, 20.23, 22.16, 20.00]
# Judgment_Time_V = [58.63, 3.96, 4.58, 5.01, 3.74]
# Assembly_Time_E = [313.67, 296.50, 437.35, 829.67, 356.58]
# Assembly_Time_V = [61.88, 48.63, 29.10, 87.55, 56.34]
# Tracking_Failure_E = [0, 0.17, 0, 3.83, 0.67]
# Tracking_Failure_V = [0, 0.37, 0, 1.34, 0.74]

# Group1_E = [665.00, 0.80, 2.33, 381.33, 313.67, 0]
# Group1_V = [76.89, 0.16, 0.47, 58.63, 61.88, 0]
# Group5_E = [313.50, 0.97, 0.50, 17.00, 296.50, 0.17]
# Group5_V = [48.34, 0.074, 0.76, 3.96, 48.63, 0.37]
# Group2_E = [457.58, 0.93, 1.00, 20.23, 437.35, 0]
# Group2_V = [30.10, 0.074, 0.58, 4.58, 29.10, 0]
# Group3_E = [851.80, 0.58, 0.83, 22.16, 829.67, 3.83]
# Group3_V = [89.48, 0.21, 0.37, 5.01, 87.55, 1.34]
# Group4_E = [376.50, 0.96, 2.16, 20.00, 356.58, 0.67]
# Group4_V = [58.19, 0.038, 1.06, 3.74, 56.34, 0.74]
# 将第一个元素向后移动两个位置
# Group1_E = [0.80, 2.33, 665.00, 381.33, 313.67, 0]
# Group1_V = [0.16, 0.47, 76.89, 58.63, 61.88, 0]
# Group5_E = [0.97, 0.50, 313.50, 17.00, 296.50, 0.17]
# Group5_V = [0.074, 0.76, 48.34, 3.96, 48.63, 0.37]
# Group2_E = [0.93, 1.00, 457.58, 20.23, 437.35, 0]
# Group2_V = [0.074, 0.58, 30.10, 4.58, 29.10, 0]
# Group3_E = [0.58, 0.83, 851.80, 22.16, 829.67, 3.83]
# Group3_V = [0.21, 0.37, 89.48, 5.01, 87.55, 1.34]
# Group4_E = [0.96, 2.16, 376.50, 20.00, 356.58, 0.67]
# Group4_V = [0.038, 1.06, 58.19, 3.74, 56.34, 0.74]

# 将最后一个元素向前移动三个位置
Group1_E = [0.80, 2.33, 0, 665.00, 381.33, 313.67]
Group1_V = [0.16, 0.47, 0, 76.89, 58.63, 61.88]
Group5_E = [0.97, 0.50, 0.17, 313.50, 17.00, 296.50]
Group5_V = [0.074, 0.76, 0.37, 48.34, 3.96, 48.63]
Group2_E = [0.93, 1.00, 0, 457.58, 20.23, 437.35]
Group2_V = [0.074, 0.58, 0, 30.10, 4.58, 29.10]
Group3_E = [0.58, 0.83, 3.83, 851.80, 22.16, 829.67]
Group3_V = [0.21, 0.37, 1.34, 89.48, 5.01, 87.55]
Group4_E = [0.96, 2.16, 0.67, 376.50, 20.00, 356.58]
Group4_V = [0.038, 1.06, 0.74, 58.19, 3.74, 56.34]

# Enjoyment_E = [2.00, 2.00, 3.33, 3.00, 4.17]
# Enjoyment_V = [0.82, 0.82, 0.47, 0.81, 0.69]
# Focus_E = [3.00, 2.50, 3.17, 3.33, 4.17]
# Focus_V = [0.82, 0.50, 0.90, 0.47, 0.89]
# Friendliness_E = [1.33, 2.66, 3.00, 3.17, 4.67]
# Friendliness_V = [0.94, 1.11, 0.58, 0.69, 0.47]
# Usability_E = [3.00, 3.00, 4.00, 3.50, 4.17]
# Usability_V = [0.41, 0.58, 0.81, 0.76, 1.21]

# y_limit = []  # 每个的最大值
# y_limit = [665.00, 0.97, 2.33, 381.33, 829, 3.83]
# y_limit = [1.00*1.5, 3.00*1.5, 700.00*1.5, 500.00, 900.00*1.5, 4.00*1.5]
y_limit = [1.00*1.5, 4.0, 4.00*1.5, 700.00*1.5, 700.00*1.5, 700.00*1.5]
y_axis_name = ['efffectiveness', 'error count(number)', 'tracking failure(times)', 'used time', 'judgment time(s)', 'assembly time(s)']

# all_attributes = ['Enjoyment', 'Focus', 'Friendliness', 'Usability']
# all_attributes = ['Efficiency', 'Effectiveness', 'Error Count', 'Judgment Time', 'Assembly Time', 'Tracking Failure']
# all_attributes = ['Group1', 'Group2', 'Group3', 'Group4', 'Group5']
all_attributes = ['Group1', 'Group2', 'Group3', 'Group4', 'Group5']
all_attributes_values = {}

# all_attributes_values['Enjoyment'] = [Enjoyment_E, Enjoyment_V]
# all_attributes_values['Focus'] = [Focus_E, Focus_V]
# all_attributes_values['Friendliness'] = [Friendliness_E, Friendliness_V]
# all_attributes_values['Usability'] = [Usability_E, Usability_V]
# all_attributes_values['Efficiency'] = [Efficiency_E, Efficiency_V]
# all_attributes_values['Effectiveness'] = [Effectiveness_E, Effectiveness_V]
# all_attributes_values['Error Count'] = [Error_Count_E, Error_Count_V]
# all_attributes_values['Judgment Time'] = [Judgment_Time_E, Judgment_Time_V]
# all_attributes_values['Assembly Time'] = [Assembly_Time_E, Assembly_Time_V]
# all_attributes_values['Tracking Failure'] = [Tracking_Failure_E, Tracking_Failure_V]

all_attributes_values['Group1'] = [Group1_E, Group1_V]
all_attributes_values['Group2'] = [Group2_E, Group2_V]
all_attributes_values['Group3'] = [Group3_E, Group3_V]
all_attributes_values['Group4'] = [Group4_E, Group4_V]
all_attributes_values['Group5'] = [Group5_E, Group5_V]


group_num = len(all_attributes_values[all_attributes[0]][0])  #  每一个list中属性的数量
groups = []
for i in range(group_num):
    g = Group('Group ' + str(i), default_group_setting)
    for attribute in all_attributes:
        g.update(attribute+"_E", all_attributes_values[attribute][0][i])
        g.update(attribute+"_V", all_attributes_values[attribute][1][i])
    groups.append(g)

# width, height
plot_width = 25  # default 10 16.5
plot_height = 9.8  # default 6.5

fig, ax = plt.subplots(figsize=(plot_width, plot_height))
plt.subplots_adjust(left=0.04, right=0.98, top=0.98, bottom=0.05)
# x = []
# group_centers = []
# bar_width = 0.25
# inner_gap = 0.05 # 0.08
# group_gap = 0.5
# group_width = len(all_attributes) * (bar_width + inner_gap) + group_gap

# ...

fig, axs = plt.subplots(1, group_num, figsize=(plot_width, plot_height))
plt.subplots_adjust(left=0.04, right=0.98, top=0.98, bottom=0.05)
# 控制子图之间的间距
plt.subplots_adjust(wspace=0.4)

ax_time = None

for i, ax in enumerate(axs):
    print(f"i: {i}/{len(axs)}")
    # if i == 3:
    #     ax_time = ax
    # if i > 3:
    #     ax = ax_time

    # 每个group单独一个subplot
    x = []
    group_centers = []
    bar_width = 0.25
    inner_gap = 0.14
    group_gap = 0.4
    group_width = len(all_attributes) * (bar_width + inner_gap) + group_gap
    
    start = i * group_width
    for j in range(len(all_attributes)):
        x.append(start +  j * (bar_width + inner_gap))
    group_centers.append(start + (len(all_attributes) * (bar_width + inner_gap) - inner_gap) / 2)

    means = []
    errors = []
    color_list = []
    group = groups[i]
# for i, group in enumerate(groups):
    means += [group.get(attribute+"_E") for attribute in all_attributes]
    errors += [group.get(attribute+"_V") for attribute in all_attributes]
    color_list += group.setting['mean_column_colors']

    ax.bar(x, means, width=bar_width, color=color_list, edgecolor='none')
    ax.set_xticks(group_centers)
    # ax.set_xticklabels([f"Group {i+1}" for i in range(group_num)], rotation=0, ha="center")
    # group_names = ["Group1", "Group2", "Group3", "Group4", "Group5"]
    group_names = ["Effectiveness↑", "Error Count↓", "Tracking Failure↓", "Total Time↓", "Judgment Time↓", "Assembly Time↓"]
    ax.set_xticklabels([group_names[i]], rotation=0, ha="center", fontweight='bold', fontfamily='Times New Roman')

    # 设置x轴字体大小
    ax.tick_params(axis='x', labelsize=default_group_setting['legend_fontsize'])
    ax.tick_params(axis='y', labelsize=default_group_setting['legend_fontsize'])
    ax.set_ylabel(y_axis_name[i], fontsize=default_group_setting['legend_fontsize'], fontfamily='Times New Roman', fontweight='bold') # 修改
    ax.tick_params(axis='y', labelsize=default_group_setting['mean_font_size'], labelcolor=default_group_setting['mean_font_color'])
    ax.errorbar(x, means, yerr=errors, fmt='k,', ecolor=default_group_setting['error_line_color'], elinewidth=1, capsize=5)

    ax.set_ylim(0, y_limit[i])  # Set the y-axis limits from 0 to 5

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    # 计算相对位置
    # mean_offset = 0.25 * (0.15/6)* y_limit[i]
    # error_offset = 0.1 * (0.15/6)* y_limit[i]
    mean_offset = 0.8 * (0.15/6)* y_limit[i]
    error_offset = 0
    # 假设这部分代码在完整代码的结尾部分，添加均值和方差标签
    for i2, (mean, error) in enumerate(zip(means, errors)):
        # error_top = mean + error - 0.1  # 计算误差线的顶端位置
        error_top = mean + error# 计算误差线的顶端位置
        if mean == 0:
            mean == "N/A"
            ax.text(x[i2], error_top + mean_offset, "N/A", ha='center', va='bottom', 
                fontsize=default_group_setting['mean_font_size'], color=tuple(default_group_setting['mean_font_color']))
            # 添加方差的标签
            ax.text(x[i2], error_top + error_offset, "N/A", ha='center', va='bottom', 
                    fontsize=default_group_setting['error_font_size'], color=tuple(default_group_setting['error_font_color']))
        else:
            ax.text(x[i2], error_top + mean_offset, f"{mean:.2f}", ha='center', va='bottom', 
                        fontsize=default_group_setting['mean_font_size'], color=tuple(default_group_setting['mean_font_color']))
            # 添加方差的标签
            ax.text(x[i2], error_top + error_offset, f"{error:.2f}", ha='center', va='bottom', 
                    fontsize=default_group_setting['error_font_size'], color=tuple(default_group_setting['error_font_color']))

# # Define color labels
# color_labels = ['Enjoyment', 'Focus', 'Friendliness', 'Usability']
# # Define the colors corresponding to each label
# color_patches = [plt.Line2D([0], [0], color=color, lw=4) for color in default_group_setting['mean_column_colors']]
# # add legend
# ax.legend(color_patches, color_labels, loc='upper left')
    if i == 0:
        print("1")
        # color_labels = ['Group1', 'Group2', 'Group3', 'Group4', 'Group5']
        color_labels = ["Blueprint", "2D Marker", "Model Fixed", "Manual Switch", "AR Guidance(Ours)"]
        xwidth = 0.15
        xgap = 0.02
        _, ax2 = plt.subplots(figsize=(plot_width, plot_height))

        # ax2 = ax.twinx()  # create second y-axis, avoid add new errorbar to origin fig
        legend_elements = [(ax2.bar(i3 - xwidth - xgap, 0, width=xwidth, label='$p(\Delta x)$', color=default_group_setting['mean_column_colors'][i3]), 
                            ax2.errorbar(i3 - xwidth - xgap, 0, yerr=0, capsize=2, elinewidth=1, fmt='k,', ecolor=default_group_setting['error_font_color'])) for i3 in range(len(color_labels))]
        # legend_elements = [(ax.bar(i - xwidth - xgap, 0, width=xwidth, label='$p(\Delta x)$', color=default_group_setting['mean_column_colors'][i])) for i in range(len(color_labels))]
        main_legend = ax.legend(legend_elements, color_labels, prop={'family': 'Times New Roman', 'weight': 'bold', 'size': default_group_setting['legend_fontsize']})
        # main_legend = ax.legend(legend_elements, color_labels, fontsize=30)
        # main_legend.set_bbox_to_anchor((0, 1, 0.3, 0.4))  # 调整图例框的位置和尺寸
        # main_legend.set_loc('upper left')  # 图例框的左中对齐

# plt.tight_layout()
plt.show()