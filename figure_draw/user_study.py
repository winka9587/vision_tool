import matplotlib.pyplot as plt

# 每一组有多个自定义属性
# 每个属性有均值和方差两个值
# 每个属性的均值和方差用柱状图和误差线表示

default_group_setting = {
    'mean_column_colors': [
        [i/255.0 for i in [156, 211, 232]],
        [i/255.0 for i in [243, 147, 147]],
        [i/255.0 for i in [251, 216, 137]],
        [i/255.0 for i in [177, 219, 158]],
        [i/255.0 for i in [135, 154, 215]]
    ],  # 默认的颜色设置
    'EV_visible': [[True, True], [True, True], [True, True], [True, True], [True, False]],  # 均值和方差是否可见
    'error_line_color': [i/255.0 for i in [56, 83, 170]],
    'mean_font_size': 12,
    'error_font_size': 12,
    'mean_font_color': [0, 0, 0],
    'error_font_color': [0.9, 0, 0.2],
    'label_offsets': [[-0.1, 0.2, 0.1], [-0.1, 0.2, 0.1], [-0.1, 0.2, 0.1], [-0.1, 0.2, 0.1], [-0.1, 0.2, 0.1]]  # 偏移量设置
    # 'label_offsets': [-0.1, 0.2, 0.1]
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

Enjoyment_E = [2.00, 4.17, 3.33, 3.00, 2.00]
Enjoyment_V = [0.82, 0.69, 0.47, 0.81, 0.82]
Focus_E = [3.00, 4.17, 3.17, 3.33, 2.50]
Focus_V = [0.82, 0.89, 0.90, 0.47, 0.50]
Friendliness_E = [1.33, 4.67, 3.00, 3.17, 2.66]
Friendliness_V = [0.94, 0.47, 0.58, 0.69, 1.11]
Usability_E = [3.00, 4.17, 4.00, 3.50, 3.00]
Usability_V = [0.41, 1.21, 0.81, 0.76, 0.58]
tmp_E = [0.7955, 0.7048, 0.8096, 0.8749, 0.7760]
tmp_V = [0,0,0,0,0]


all_attributes = ['Enjoyment', 'Focus', 'Friendliness', 'Usability', 'tmp']
all_attributes_values = {}

all_attributes_values['Enjoyment'] = [Enjoyment_E, Enjoyment_V]
all_attributes_values['Focus'] = [Focus_E, Focus_V]
all_attributes_values['Friendliness'] = [Friendliness_E, Friendliness_V]
all_attributes_values['Usability'] = [Usability_E, Usability_V]
all_attributes_values['tmp'] = [tmp_E, tmp_V]

group_num = 5
groups = []
for i in range(group_num):
    g = Group('Group ' + str(i), default_group_setting)
    for attribute in all_attributes:
        g.update(attribute+"_E", all_attributes_values[attribute][0][i])
        g.update(attribute+"_V", all_attributes_values[attribute][1][i])
    groups.append(g)

fig, ax = plt.subplots(figsize=(10, 10))

x = []
group_centers = []
bar_width = 0.2
inner_gap = 0.08
group_gap = 0.3
group_width = len(all_attributes) * (bar_width + inner_gap) + group_gap

for i in range(group_num):
    start = i * group_width
    for j in range(len(all_attributes)):
        x.append(start + j * (bar_width + inner_gap))
    group_centers.append(start + (len(all_attributes) * (bar_width + inner_gap) - inner_gap) / 2)

means = []
errors = []
color_list = []

# 绘制柱状图和可选的误差线
for i, group in enumerate(groups):
    visible_settings = group.setting['EV_visible'][i]  # 获取当前组的可见设置
    label_offsets = group.setting['label_offsets']  # 获取文本偏移设置

    means = [group.get(attribute+"_E") for attribute in all_attributes]
    errors = [group.get(attribute+"_V") for attribute in all_attributes]
    color_list = group.setting['mean_column_colors']

    # 绘制柱状图
    ax.bar(x, means, width=bar_width, color=color_list, edgecolor='none')

    # 绘制误差线，如果可见
    if visible_settings[0] and visible_settings[1]:
        ax.errorbar(x, means, yerr=errors, fmt='k,', ecolor=default_group_setting['error_line_color'], elinewidth=1, capsize=3)

    # 添加均值和方差标签，如果可见
    for j, (mean, error) in enumerate(zip(means, errors)):
        error_top = mean + error + label_offsets[0]  # 计算误差线的顶端位置，应用偏移量
        if visible_settings[0]:  # 如果均值可见
            ax.text(x[j], error_top + label_offsets[1], f"{mean:.2f}", ha='center', va='bottom', 
                    fontsize=default_group_setting['mean_font_size'], color=tuple(default_group_setting['mean_font_color']))
        if visible_settings[1]:  # 如果方差可见
            ax.text(x[j], error_top + label_offsets[2], f"±{error:.2f}", ha='center', va='bottom', 
                    fontsize=default_group_setting['error_font_size'], color=tuple(default_group_setting['error_font_color']))

plt.tight_layout()
plt.show()
