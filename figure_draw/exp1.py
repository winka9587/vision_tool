import matplotlib.pyplot as plt

default_group_setting = {
    'mean_column_colors': [
        [i/255.0 for i in [126, 153, 244]],
        [i/255.0 for i in [204, 124, 113]],
        [i/255.0 for i in [122, 182, 83]],
        [i/255.0 for i in [219, 180, 40]]
    ],
    'EV_visible': [[True, True], [True, True], [True, True], [True, True]],
    # 'error_line_color': [i/255.0 for i in [189, 72, 53]],
    'error_line_color': [i/255.0 for i in [144, 55, 40]],
    # 144, 55, 40 / 255.0 结果是 [0.5647058823529412, 0.21568627450980393, 0.1568627450980392] 保留2位小数是 [0.56, 0.22, 0.16]
    'mean_font_size': 12,
    'error_font_size': 12,
    'legend_fontsize': 16, 
    'mean_font_color': [0, 0, 0],
    'error_font_color': [0.9, 0, 0.2]
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

group_names = ["Blueprint (0.7955)", "2D Marker (0.8096)", "Model-Invariant (0.8749)", "Manual Switching (0.7760)", "AR Guidance(Ours) (0.7048)"]
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

plot_width = 15.5
plot_height = 6 # 6.5
# plot_width = 12
# plot_height = 6.5
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

means = []
errors = []
color_list = []

# 绘制两段柱状图
for i, group in enumerate(groups):
    for j, attribute in enumerate(all_attributes):
        base_x = x[i * len(all_attributes) + j]
        E_value = group.get(attribute+"_E")
        Min_value = globals()[attribute + "_Min"][i]
        Max_value = globals()[attribute + "_Max"][i]
        
        # 计算 E_value 到 Min_value 和 Max_value 的距离
        distance_to_min = abs(E_value - Min_value)
        distance_to_max = abs(E_value - Max_value)
        
        # 如果 mean 更接近 Min，则 E 到 Max 部分透明度为 0.8
        if distance_to_min <= distance_to_max:
            # 第一段: Min 到 E (透明度为默认值)
            ax.bar(base_x, E_value - Min_value, bottom=Min_value, width=bar_width, 
                   color=group.setting['mean_column_colors'][j], alpha=1, edgecolor='none')

            # 第二段: E 到 Max (透明度为 0.8)
            ax.bar(base_x, Max_value - E_value, bottom=E_value, width=bar_width, 
                   color=group.setting['mean_column_colors'][j], alpha=0.6, edgecolor='none')
        else:
            # 如果 mean 更接近 Max，则 Min 到 E 部分透明度为 0.8
            # 第一段: Min 到 E (透明度为 0.8)
            ax.bar(base_x, E_value - Min_value, bottom=Min_value, width=bar_width, 
                   color=group.setting['mean_column_colors'][j], alpha=0.6, edgecolor='none')

            # 第二段: E 到 Max (透明度为默认值)
            ax.bar(base_x, Max_value - E_value, bottom=E_value, width=bar_width, 
                   color=group.setting['mean_column_colors'][j], alpha=1, edgecolor='none')


        means.append(E_value)
        errors.append(group.get(attribute+"_V"))

ax.set_xticks(group_centers)
ax.set_xticklabels(group_names, rotation=0, ha="center", fontweight='bold', fontfamily='Times New Roman')

ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.set_ylabel('Score', fontsize=16, fontfamily='Times New Roman', fontweight='bold')
ax.set_ylim(0, 6.9)
ax.set_yticks([i for i in range(0, 7) if i != 6])  # 不显示6

# 计算误差线的中心点 (Min + Max) / 2
center_points = [(globals()[attribute + "_Min"][i] + globals()[attribute + "_Max"][i]) / 2 for i in range(group_num) for attribute in all_attributes]

extended_errors = [error + 0.8 for error in errors]  # 增加0.1来扩展误差线的长度
# 绘制误差线，中心点为 (Min + Max) / 2，误差线长为 (Max - Min) / 2
ax.errorbar(x, center_points, yerr=extended_errors, fmt='k,', ecolor=default_group_setting['error_line_color'], elinewidth=1, capsize=8)

# 添加标签，注意现在基于中心点来计算误差线的标签位置
for i, (center, error) in enumerate(zip(center_points, extended_errors)):
    error_top = center + error - 0.1
    ax.text(x[i], error_top + 0.3, f"{center:.2f}", ha='center', va='bottom', 
            fontsize=default_group_setting['mean_font_size'], color=tuple(default_group_setting['mean_font_color']))
    ax.text(x[i], error_top + 0.1, f"{error:.2f}", ha='center', va='bottom', 
            fontsize=default_group_setting['error_font_size'], color=tuple(default_group_setting['error_font_color']))

# 添加legend

# color_labels = ['Enjoyment Mean and S.D.', 'Focus Mean and S.D.', 'Friendliness Mean and S.D.', 'Usability Mean and S.D.']
color_labels = ['Enjoyment', 'Focus', 'Friendliness', 'Usability']
xwidth = 0.15
xgap = 0.02
_, ax2 = plt.subplots(figsize=(plot_width, plot_height))

# ax2 = ax.twinx()  # create second y-axis, avoid add new errorbar to origin fig
legend_elements = [(ax2.bar(i - xwidth - xgap, 0, width=xwidth, label='$p(\Delta x)$', color=default_group_setting['mean_column_colors'][i]), 
                    ax2.errorbar(i - xwidth - xgap, 0, yerr=0, capsize=2, elinewidth=1, fmt='k,', ecolor=default_group_setting['error_font_color'])) for i in range(len(color_labels))]

# legend_elements = [(ax.bar(i - xwidth - xgap, 0, width=xwidth, label='$p(\Delta x)$', color=default_group_setting['mean_column_colors'][i])) for i in range(len(color_labels))]
main_legend = ax.legend(legend_elements, color_labels, prop={'family': 'Times New Roman', 'weight': 'bold', 'size': default_group_setting['legend_fontsize']})

plt.show()
