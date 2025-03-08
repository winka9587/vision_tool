
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman') 
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from matplotlib.pyplot import MultipleLocator
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats import laplace
from scipy import stats

plt.rcParams['font.family'] = 'Times New Roman'

class Case:
    def __init__(self, id, name, gender, age, ar_exp, model, tx, ty, tz, r):
        self.id = id
        self.name = name
        self.gender = gender
        self.age = age
        self.ar_exp = ar_exp
        self.model = model
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.r = r 
        
    def __str__(self):
            return f"ID: {self.id}, Name: {self.name}, Gender: {self.gender}, Age: {self.age}, AR Exp: {self.ar_exp}, Model: {self.model}, Tx: {self.tx}, Ty: {self.ty}, Tz: {self.tz}, R: {self.r}"

def abs_case(case):
    abs_tx = np.abs(case.tx)
    abs_ty = np.abs(case.ty)
    abs_tz = np.abs(case.tz)
    abs_r = np.abs(case.r)
    
    return Case(case.id, case.name, case.gender, case.age, case.ar_exp, case.model, abs_tx, abs_ty, abs_tz, abs_r)

    

def find_cases_by_model(cases, target_model):
    result = []
    for case in cases:
        if case.model == target_model:
            # print("case", case)
            result.append(case)
    return result

class StatisticData:
    def __init__(self, tx_mean, ty_mean, tz_mean, r_mean, tx_std, ty_std, tz_std, r_std):
        self.tx_mean = tx_mean
        self.ty_mean = ty_mean
        self.tz_mean = tz_mean
        self.r_mean = r_mean 
        self.tx_std = tx_std
        self.ty_std = ty_std
        self.tz_std = tz_std
        self.r_std = r_std 
    
    def print_info(self, words):
        print(words, ' Tx Mean Std {:.4f} {:.4f}'.format(self.tx_mean, self.tx_std))
        print(words, ' Ty Mean Std {:.4f} {:.4f}'.format(self.ty_mean, self.ty_std))
        print(words, ' Tz Mean Std {:.4f} {:.4f}'.format(self.tz_mean, self.tz_std))
        print(words, ' R  Mean Std {:.4f} {:.4f}'.format(self.r_mean, self.r_std))
    
    def print_tx_info(self, words):
        print(words, ' Tx Mean Std {:.4f} {:.4f}'.format(self.tx_mean, self.tx_std))
    def print_ty_info(self, words):
        print(words, ' Ty Mean Std {:.4f} {:.4f}'.format(self.ty_mean, self.ty_std))
    def print_tz_info(self, words):
        print(words, ' Tz Mean Std {:.4f} {:.4f}'.format(self.tz_mean, self.tz_std))
    def print_R_info(self, words):
        print(words, ' R  Mean Std {:.4f} {:.4f}'.format(self.r_mean, self.r_std))

# 定义一个函数来计算均值和标准差
def calculate_stats(cases, info):
    # 初始化列表来存储每个参数的值
    txs = []
    tys = []
    tzs = []
    rs = []

    # 遍历案例，收集参数值
    for case in cases:
        # print(case)
        txs.append(case.tx)
        tys.append(case.ty)
        tzs.append(case.tz)
        rs.append(case.r)

    # 计算均值
    tx_mean = np.mean(txs)
    ty_mean = np.mean(tys)
    tz_mean = np.mean(tzs)
    r_mean = np.mean(rs)

    # 计算标准差
    tx_std = np.std(txs, ddof=1)
    ty_std = np.std(tys, ddof=1)
    tz_std = np.std(tzs, ddof=1)
    r_std = np.std(rs, ddof=1)

    print('\n'+info+'tx的均值和标准差: {:.3f} {:.3f}'.format(tx_mean, tx_std), \
          '\n'+info+'ty的均值和标准差: {:.3f} {:.3f}'.format(ty_mean, ty_std), \
          '\n'+info+'tz的均值和标准差: {:.3f} {:.3f}'.format(tz_mean, tz_std), \
          '\n'+info+'r的均值和标准差:  {:.1f} {:.1f}'.format(r_mean, r_std))
	
    return StatisticData(tx_mean, ty_mean, tz_mean, r_mean, tx_std, ty_std, tz_std, r_std)
  
user_1 = {'id': 1, 'name': 'cx',  'gender': 'M', 'age': 26, 'ar_exp': True}
user_2 = {'id': 2, 'name': 'kf',  'gender': "F", 'age': 25, 'ar_exp': True}
user_3 = {'id': 3, 'name': 'zyf', 'gender': "M", 'age': 24, 'ar_exp': True}
user_4 = {'id': 4, 'name': 'xk',  'gender': "M", 'age': 24, 'ar_exp': True}
user_5 = {'id': 5, 'name': 'lj',  'gender': "M", 'age': 23, 'ar_exp': True}
user_6 = {'id': 6, 'name': 'zpp', 'gender': "M", 'age': 25,'ar_exp': False}
user_7 = {'id': 7, 'name': 'lsy', 'gender': "M", 'age': 25, 'ar_exp': True}
user_8 = {'id': 8, 'name': 'swb', 'gender': "M", 'age': 22, 'ar_exp': True}
user_9 = {'id': 9, 'name': 'zzx', 'gender': "M", 'age': 23, 'ar_exp': True}
user_10 = {'id':10, 'name': 'sxq', 'gender': "M", 'age': 27, 'ar_exp': True}
user_11 = {'id':11, 'name': 'scx', 'gender': "M", 'age': 28, 'ar_exp': True}
user_12 = {'id':12, 'name': 'qxy', 'gender': "F", 'age': 57, 'ar_exp': True}	
user_13 = {'id':13, 'name': 'lx', 'gender': "M", 'age': 26, 'ar_exp': True}	
user_14 = {'id':14, 'name': 'cx', 'gender': "M", 'age': 27, 'ar_exp': False}
user_15 = {'id':15, 'name': 'fj', 'gender': "F", 'age': 23, 'ar_exp': True}
user_16 = {'id':16, 'name': 'wcs', 'gender': "M", 'age': 24, 'ar_exp': True}
users = [user_1,  user_2,  user_3,  user_4,  user_5,  user_6, user_7, user_8, user_9, user_10, \
         user_11, user_12, user_13, user_14, user_15, user_16]     

def read_data_from_txt(file_path):
    data = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split('\t')
        model = parts[0]
        tx = float(parts[1])
        ty = float(parts[2])
        tz = float(parts[3])
        r = float(parts[4])
        data.append((model, tx, ty, tz, r))

    return data

root_path = "/home/songxiuqiang/dataset/realtime/realsense/24tvcg/error_perception/user_data/txt/"

obj_nums = 7

file_1 = root_path + '1.txt'
file_2 = root_path + '2.txt'
file_3 = root_path + '3.txt'
file_4 = root_path + '4.txt'
file_5 = root_path + '5.txt'
file_6 = root_path + '6.txt'
file_7 = root_path + '7.txt'
file_8 = root_path + '8.txt'
file_9 = root_path + '9.txt'
file_10 = root_path + '10.txt'
file_11 = root_path + '11.txt'
file_12 = root_path + '12.txt'
file_13 = root_path + '13.txt'
file_14 = root_path + '14.txt'
file_15 = root_path + '15.txt'
file_16 = root_path + '16.txt'
files  = [file_1,  file_2,  file_3,  file_4,  file_5,  file_6, file_7, file_8, file_9, file_10, \
          file_11, file_12, file_13, file_14, file_15, file_16]

file_1_r = root_path + '1_r.txt'
file_2_r = root_path + '2_r.txt'
file_3_r = root_path + '3_r.txt'
file_4_r = root_path + '4_r.txt'
file_5_r = root_path + '5_r.txt'
file_6_r = root_path + '6_r.txt'
file_7_r = root_path + '7_r.txt'
file_8_r = root_path + '8_r.txt'
file_9_r = root_path + '9_r.txt'
file_10_r = root_path + '10_r.txt'
file_11_r = root_path + '11_r.txt'
file_12_r = root_path + '12_r.txt'
file_13_r = root_path + '13_r.txt'
file_14_r = root_path + '14_r.txt'
file_15_r = root_path + '15_r.txt'
file_16_r = root_path + '16_r.txt'
files_r  = [file_1_r,  file_2_r,  file_3_r,  file_4_r,  file_5_r,  file_6_r, file_7_r, file_8_r, file_9_r, file_10_r, \
            file_11_r, file_12_r, file_13_r, file_14_r, file_15_r, file_16_r]


datas = []
datas_r = []
cases = []
cases_r = []

print("users num", len(users))
for i in range (0,len(users)):
    datas.append(read_data_from_txt(files[i]))
    datas_r.append(read_data_from_txt(files_r[i]))

for i in range (0,len(users)):
    data = datas[i]	# 一名用户的所有数据
    data_r = datas_r[i]
    for item in data:
        case = Case(**users[i], model=item[0], tx=item[1], ty=item[2], tz=item[3], r=item[4])
        cases.append(case)
        cases_r.append(abs_case(case))
    for item in data_r: 
        case_r = Case(**users[i], model=item[0], tx=item[1], ty=item[2], tz=item[3], r=item[4])
        cases_r.append(case_r)

cases_ape = find_cases_by_model(cases,"ape")        
cases_cat = find_cases_by_model(cases,"cat")           
cases_s_s = find_cases_by_model(cases,"s-small")           
cases_s_l = find_cases_by_model(cases,"s-large")           
cases_a_R = find_cases_by_model(cases,"ape-rbot")           
cases_c_R = find_cases_by_model(cases,"cat-rbot")           
cases_s_R = find_cases_by_model(cases,"s-rbot")  
 
cases_ape_r = find_cases_by_model(cases_r,"ape")        
cases_cat_r = find_cases_by_model(cases_r,"cat")           
cases_s_s_r = find_cases_by_model(cases_r,"s-small")           
cases_s_l_r = find_cases_by_model(cases_r,"s-large")           
cases_a_R_r = find_cases_by_model(cases_r,"ape-rbot")           
cases_c_R_r = find_cases_by_model(cases_r,"cat-rbot")           
cases_s_R_r = find_cases_by_model(cases_r,"s-rbot")  

ape_data = calculate_stats(cases_ape, "ape      正向 ")
cat_data = calculate_stats(cases_cat, "cat      正向 ")
s_s_data = calculate_stats(cases_s_s, "s_small  正向 ")
s_l_data = calculate_stats(cases_s_l, "s_large  正向 ")
a_R_data = calculate_stats(cases_a_R, "ape_rbot 正向 ")
c_R_data = calculate_stats(cases_c_R, "cat_rbot 正向 ")
s_R_data = calculate_stats(cases_s_R, "s_rbot   正向 ")

ape_data_r = calculate_stats(cases_ape_r, "ape      反向 ")
cat_data_r = calculate_stats(cases_cat_r, "cat      反向 ")
s_s_data_r = calculate_stats(cases_s_s_r, "s_small  反向 ")
s_l_data_r = calculate_stats(cases_s_l_r, "s_large  反向 ")
a_R_data_r = calculate_stats(cases_a_R_r, "ape_rbot 反向 ")
c_R_data_r = calculate_stats(cases_c_R_r, "cat_rbot 反向 ")
s_R_data_r = calculate_stats(cases_s_R_r, "s_rbot   反向 ")

calculate_stats(cases_r, "All models ")

model_datas = [ape_data, cat_data, s_s_data, s_l_data, a_R_data, c_R_data, s_R_data]
model_datas_r = [ape_data_r, cat_data_r, s_s_data_r, s_l_data_r, a_R_data_r, c_R_data_r, s_R_data_r]

tx_means = []
ty_means = []
tz_means = []
tx_stds = []
ty_stds = []
tz_stds = []
r_means = []
r_stds = []


tx_means_r = []
ty_means_r = []
tz_means_r = []
tx_stds_r = []
ty_stds_r = []
tz_stds_r = []
r_means_r = []
r_stds_r = []


for model_data in model_datas:
    tx_means.append(model_data.tx_mean)
    ty_means.append(model_data.ty_mean)
    tz_means.append(model_data.tz_mean)
    r_means.append(model_data.r_mean)
    tx_stds.append(model_data.tx_std)
    ty_stds.append(model_data.ty_std)
    tz_stds.append(model_data.tz_std)
    r_stds.append(model_data.r_std)

for model_data_r in model_datas_r:
    tx_means_r.append(model_data_r.tx_mean)
    ty_means_r.append(model_data_r.ty_mean)
    tz_means_r.append(model_data_r.tz_mean)
    r_means_r.append(model_data_r.r_mean)
    tx_stds_r.append(model_data_r.tx_std)
    ty_stds_r.append(model_data_r.ty_std)
    tz_stds_r.append(model_data_r.tz_std)
    r_stds_r.append(model_data_r.r_std)
    
    
def column_x_y(x,var1,var2,b1,b2):
    width=0.2
    xa=np.arange(len(x))
    
    plt.figure(figsize=(6,4))
    plt.bar(xa-width/2,var1,width=width,label='normoxia')    #绘制柱形图
    plt.errorbar(xa-width/2,var1,yerr=b1,capsize=2,elinewidth=1,fmt='k,')  #绘制标准差

    plt.bar(xa+width/2,var2,width=width,label='hypoxia')
    plt.errorbar(xa+width/2,var2,yerr=b2,capsize=2,elinewidth=1,fmt='k,')

    plt.xticks(xa,x)
    plt.legend(ncol=2)

    plt.show()


def column_x_y_z(ax, x, var1, var2, var3, b1, b2, b3):
    gap = 0.05  # 设置长条之间的间隔
    width = 0.2
    xa = np.arange(len(x))
    
    color_1 = (126 / 255, 153 / 255, 244 / 255)
    color_2 = (204 / 255, 124 / 255, 113 / 255)
    color_3 = (122 / 255, 182 / 255, 83 / 255)
    color_4 = (192 / 255, 50 / 255,  26 / 255)
    
    bars1 = ax.bar(xa - width - gap, var1, width=width, label='$p(\Delta x)$', color=color_1)    # 绘制第一组柱形图，并设置颜色为蓝色
    eb1 = ax.errorbar(xa - width - gap, var1, yerr=b1, capsize=2, elinewidth=1, fmt='k,', ecolor=color_4)  # 绘制第一组标准差，并设置颜色为红色

    bars2 = ax.bar(xa, var2, width=width, label='$p(\Delta y)$', color=color_2)  # 绘制第二组柱形图，并设置颜色为绿色
    eb2 = ax.errorbar(xa, var2, yerr=b2, capsize=2, elinewidth=1, fmt='k,', ecolor=color_4)  # 绘制第二组标准差，并设置颜色为红色

    bars3 = ax.bar(xa + width + gap, var3, width=width, label='$p(\Delta z)$', color=color_3)  # 绘制第三组柱形图，并设置颜色为橙色
    eb3 = ax.errorbar(xa + width + gap, var3, yerr=b3, capsize=2, elinewidth=1, fmt='k,', ecolor=color_4)  # 绘制第三组标准差，并设置颜色为红色

    # 在每个长条的顶部显示均值和标准差
    x_off_1 = [0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09]
    y_off_1 = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]

    x_off_2 = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
    y_off_2 = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]

    x_off_3 = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
    y_off_3 = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
    off_id_1 = 0
    off_id_2 = 0
    off_id_3 = 0
    for bar, mean, std in zip(bars1, var1, b1):
        ax.text(bar.get_x() + bar.get_width() / 2 - x_off_1[off_id_1], bar.get_height() + std + 0.01  + y_off_1[off_id_1], f'{mean:.2f}', ha='center', va='bottom', fontsize=10, color='black')
        ax.text(bar.get_x() + bar.get_width() / 2 - x_off_1[off_id_1], bar.get_height() + std + 0.01 , f'{std:.2f}', ha='center', va='bottom', fontsize=10, color='red')
        off_id_1 = off_id_1 + 1

    for bar, mean, std in zip(bars2, var2, b2):
        ax.text(bar.get_x() + bar.get_width() / 2 - x_off_2[off_id_2], bar.get_height() + std + 0.01 + y_off_2[off_id_2], f'{mean:.2f}', ha='center', va='bottom', fontsize=10, color='black')
        ax.text(bar.get_x() + bar.get_width() / 2 - x_off_2[off_id_2], bar.get_height() + std + 0.01, f'{std:.2f}', ha='center', va='bottom', fontsize=10, color='red')
        off_id_2 = off_id_2 + 1

    for bar, mean, std in zip(bars3, var3, b3):
        ax.text(bar.get_x() + bar.get_width() / 2 - x_off_3[off_id_3], bar.get_height() + std + 0.01 + y_off_3[off_id_3], f'{mean:.2f}', ha='center', va='bottom', fontsize=10, color='black')
        ax.text(bar.get_x() + bar.get_width() / 2 - x_off_3[off_id_3], bar.get_height() + std + 0.01, f'{std:.2f}', ha='center', va='bottom', fontsize=10, color='red')
        off_id_3 = off_id_3 + 1

    ax.set_xticks(xa)
    ax.set_xticklabels(x)
    
    ax.set_ylabel('Proportion of The BBX Diagonal Length', fontsize=14)
    
    # ax.legend(ncol=3, fontsize=14)
    legend_elements = [(bars1, eb1), (bars2, eb2), (bars3, eb3)]
    labels = ["Error $p(\Delta x)$ Mean and S.D.", "Error $p(\Delta y)$ Mean and S.D.", "Error $p(\Delta z)$ Mean and S.D."]
    ax.legend(legend_elements, labels, fontsize=14)
    # ax.legend(labels, prop={'style': 'italic'})
    
    plt.subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.1)
    plt.ylim(-0.02, 0.65)
    
def column_R(ax, x, mean, std):
    width = 0.3
    xa = np.arange(len(x))
    
    color_5 = (219 / 255, 180 /255, 40 / 255)
    color_4 = (192 / 255, 50 / 255,  26 / 255) 

    ax.bar(xa, mean, width=width, label='Perceptible rotation error $p(\Delta R)$ to users', color=color_5)    
    ax.errorbar(xa, mean, yerr=std, capsize=2, elinewidth=1.5, fmt='k,', ecolor=color_4) 

    ax.set_xticks(xa)
    ax.set_xticklabels(x)
    # ax.set_xlabel('Model')  
    ax.set_ylabel('degree')  
    ax.legend(ncol=1, fontsize=14)

  
def column_x_y_z_R(ax, x, var1, var2, var3, var4, b1, b2, b3, b4):
    gap = 0.05  # 设置长条之间的间隔
    width = 0.15
    xa = np.arange(len(x))
    
    color_1 = (126 / 255, 153 / 255, 244 / 255)
    color_2 = (204 / 255, 124 / 255, 113 / 255)
    color_3 = (122 / 255, 182 / 255, 83 / 255)
    color_4 = (192 / 255, 50 / 255,  26 / 255)
    color_5 = (219 / 255, 180 /255,  40 / 255)
    
    bars1 = ax.bar(xa - width - gap, var1, width=width, label='$p(\Delta x)$', color=color_1)    # 绘制第一组柱形图，并设置颜色为蓝色
    eb1 = ax.errorbar(xa - width - gap, var1, yerr=b1, capsize=2, elinewidth=1, fmt='k,', ecolor=color_4)  # 绘制第一组标准差，并设置颜色为红色

    bars2 = ax.bar(xa, var2, width=width, label='$p(\Delta y)$', color=color_2)  # 绘制第二组柱形图，并设置颜色为绿色
    eb2 = ax.errorbar(xa, var2, yerr=b2, capsize=2, elinewidth=1, fmt='k,', ecolor=color_4)  # 绘制第二组标准差，并设置颜色为红色

    bars3 = ax.bar(xa + width + gap, var3, width=width, label='$p(\Delta z)$', color=color_3)  # 绘制第三组柱形图，并设置颜色为橙色
    eb3 = ax.errorbar(xa + width + gap, var3, yerr=b3, capsize=2, elinewidth=1, fmt='k,', ecolor=color_4)  # 绘制第三组标准差，并设置颜色为红色

    # Create a secondary y-axis for var4
    ax2 = ax.twinx()
    bars4 = ax2.bar(xa + 2*width + 2*gap, var4, width=width, label='Perceptible rotation error $p(\Delta R)$ to users', color=color_5)
    eb4 = ax2.errorbar(xa + 2*width + 2*gap, var4, yerr=b4, capsize=2, elinewidth=1, fmt='k,', ecolor=color_4)
    
    not_size = 12
    
    # 在每个长条的顶部显示均值和标准差
    for bar, mean, std in zip(bars1, var1, b1):
        ax.text(bar.get_x()+bar.get_width()/2-0.00, bar.get_height()+std + 0.035, f'{mean:.2f}', ha='center', va='bottom', fontsize=not_size, color='black')
        ax.text(bar.get_x()+bar.get_width()/2-0.00, bar.get_height()+std + 0.010, f'{std:.2f}', ha='center', va='bottom', fontsize=not_size, color='red')

    for bar, mean, std in zip(bars2, var2, b2):
        ax.text(bar.get_x()+bar.get_width()/2-0.00, bar.get_height()+std + 0.035, f'{mean:.2f}', ha='center', va='bottom', fontsize=not_size, color='black')
        ax.text(bar.get_x()+bar.get_width()/2-0.00, bar.get_height()+std + 0.010, f'{std:.2f}', ha='center', va='bottom', fontsize=not_size, color='red')

    for bar, mean, std in zip(bars3, var3, b3):
        ax.text(bar.get_x()+bar.get_width()/2-0.00, bar.get_height()+std + 0.035, f'{mean:.2f}', ha='center', va='bottom', fontsize=not_size, color='black')
        ax.text(bar.get_x()+bar.get_width()/2-0.00, bar.get_height()+std + 0.010, f'{std:.2f}', ha='center', va='bottom', fontsize=not_size, color='red')

    for bar, mean, std in zip(bars4, var4, b4):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 3.5, f'{mean:.2f}', ha='center', va='bottom', fontsize=not_size, color='black')
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 1.0, f'{std:.2f}', ha='center', va='bottom', fontsize=not_size, color='red')

    ax.set_xticks(xa+0.07)
    ax.set_xticklabels(x)
    
    # Set labels for both y-axes
    ax.set_ylabel('Translation Error \n (Absolute Error / BBX Diagonal Length)', fontsize=16, color = 'blue')
    ax2.set_ylabel('Rotation Error (Degree)', fontsize=18, color = 'darkorange')
    
    ax.set_ylim(-0.02, 0.65)  # Adjust as needed
    ax2.set_ylim(-2, 70)   # Adjust as needed for appropriate scale
    
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16) 
    ax2.tick_params(axis='y', labelsize=16) 
    
    legend_elements = [(bars1, eb1), (bars2, eb2), (bars3, eb3), (bars4, eb4)]
    labels = ["Translation Error $p(\Delta x)$ Mean and S.D.", \
              "Translation Error $p(\Delta y)$ Mean and S.D.", \
              "Translation Error $p(\Delta z)$ Mean and S.D.", \
              "Rotation Error $p(\Delta R)$ Mean and S.D."]
    ax.legend(legend_elements, labels, fontsize=14)
    
    plt.subplots_adjust(left=0.06, right=0.96, top=0.98, bottom=0.1)

# 创建一个包含两个子图的图形
# fig, (ax, ax) = plt.subplots(1, 2, figsize=(12, 6))
fig, ax = plt.subplots(figsize=(18, 6))

x_axis = ['Ape (Real)', 'Cat (Real)',' Small Squi (Real)', 'Squi (Real)', 'Ape (RBOT)', 'Cat (RBOT)', 'Squi (RBOT)']
# x_axis = ['Ape', 'Cat',' Small Squi', 'Squi', 'Ape (RBOT)', 'Cat (RBOT)', 'Squi (RBOT)']
# column_x_y_z(ax, x_axis, tx_means_r, ty_means_r, tz_means_r, tx_stds_r, ty_stds_r, tz_stds_r)
# column_R(ax,x_axis,r_means_r, r_stds_r)
column_x_y_z_R(ax, x_axis, \
             tx_means_r, ty_means_r, tz_means_r, r_means_r,\
             tx_stds_r,  ty_stds_r,  tz_stds_r,  r_stds_r)

# 调整标注的字体大小
# ax.tick_params(axis='x', labelsize=14)
# ax.tick_params(axis='y', labelsize=14)

plt.show()