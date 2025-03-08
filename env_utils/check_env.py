# 使用方法:
# 将该py文件移动到与environment.yml同级目录下
# 激活目标环境后，运行 python check_env.py
# 本文件将会检查当前激活环境与environment.yml中指定的库差异
# 未安装的库将会以 [NO] 标记，已安装的库将会以 [OK] 标记
# 若已安装的库版本高于environment.yml中指定的版本，将会以绿色标记
# 若已安装的库版本低于environment.yml中指定的版本，将会以红色标记
# 若environment.yml中未指定版本，将会以默认颜色标记

# 注意事项:
# cudatoolkit 与 cuda-cupti等cuda相关库的功能性差异不会被检查, 需手动测试

#!/usr/bin/env python3
import subprocess
import re
import yaml
import os

try:
    from packaging.version import parse as parse_version
except ImportError:
    parse_version = None  # 如果未安装 packaging，则无法进行版本比较

def normalize_pip_path(pip_dep_str):
    """
    针对 'pip:' 下的依赖进行简易处理:
    1) 若包含类似 '../some/dir/pkg_name' 或 'submodules/pkg_name'，取最后一段 'pkg_name'。
    2) 若还带版本符号（如 'pkg_name==1.2.3' 或 'submodules/pkg_name==1.2.3'），保留版本信息分开处理。
    """
    # 先把形如 'some/path/pkg_name==1.2.3' 中的路径部分去掉
    # os.path.split('some/path/pkg_name==1.2.3') -> ('some/path', 'pkg_name==1.2.3')
    _, filename = os.path.split(pip_dep_str.strip())
    # 这时 filename 可能是 'pkg_name==1.2.3' 或 'pkg_name', 'submodules' 等

    # 再分离出版本信息
    parts = re.split(r'[=>]+', filename)
    pkg_name_raw = parts[0].strip()   # 没有去掉版本号的纯字符串（不含 ==1.2.3）
    pkg_version = None
    if len(parts) > 1:
        pkg_version = parts[1].strip()

    # 如果 pkg_name_raw 里还包含可能的 '.git'、'.whl'、或其他后缀，你也可以做进一步处理
    # 这里示例中简单返回 pkg_name_raw、pkg_version
    return pkg_name_raw, pkg_version

def parse_environment_yml(yml_path):
    env_deps_list = []
    with open(yml_path, "r", encoding="utf-8") as f:
        env_yml = yaml.safe_load(f)

    for dep in env_yml.get("dependencies", []):
        if isinstance(dep, str):
            # 例: "numpy==1.25.1" 或 "numpy=1.25.1"
            parts = re.split(r'[=>]+', dep)
            pkg_name = parts[0].strip()
            pkg_version = None
            if len(parts) > 1:
                pkg_version = parts[1].strip()
            env_deps_list.append((pkg_name, pkg_version))
        elif isinstance(dep, dict) and "pip" in dep:
            # 处理 pip: [ ... ] 列表
            for pip_dep in dep["pip"]:
                pkg_name_raw, pkg_version = normalize_pip_path(pip_dep)
                env_deps_list.append((pkg_name_raw, pkg_version))

    return env_deps_list


def get_installed_packages():
    """
    分别通过 'conda list' 和 'pip list' 获取已安装的包及版本，返回字典:
      installed = {
        'numpy': '1.25.2',
        'pillow': '10.3.0',
        ...
      }
    注意：用小写做 key，以便忽略大小写。
    """
    installed = {}

    # 1) conda list
    conda_proc = subprocess.run(["conda", "list"], capture_output=True, text=True)
    for line in conda_proc.stdout.splitlines():
        # 跳过注释或空行
        if line.startswith("#") or not line.strip():
            continue
        tokens = line.split()
        if len(tokens) >= 2:
            pkg_name = tokens[0].lower()
            pkg_version = tokens[1]
            installed[pkg_name] = pkg_version

    # 2) pip list
    pip_proc = subprocess.run(["pip", "list"], capture_output=True, text=True)
    lines = pip_proc.stdout.splitlines()
    # 前2行通常是表头 "Package    Version"
    for line in lines[2:]:
        tokens = line.split()
        if len(tokens) >= 2:
            pkg_name = tokens[0].lower()
            pkg_version = tokens[1]
            installed[pkg_name] = pkg_version

    return installed

# 简单的终端颜色前缀 / 后缀
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

def compare_versions(current_ver, env_ver):
    """
    比较当前版本 current_ver 与 environment.yml中版本 env_ver:
      - current_ver > env_ver => ">"
      - current_ver = env_ver => "="
      - current_ver < env_ver => "<"
      - 任一为空 或 parse_version 不可用 => None
    """
    if not parse_version or not current_ver or not env_ver:
        return None

    try:
        iv = parse_version(current_ver)
        ev = parse_version(env_ver)
        if iv > ev:
            return ">"
        elif iv < ev:
            return "<"
        else:
            return "="
    except:
        return None

def main():
    yml_path = "environment.yml"  # 修改为实际 environment.yml 路径
    env_deps_list = parse_environment_yml(yml_path)
    installed_pkgs = get_installed_packages()

    installed_list = []
    not_installed_list = []

    # 遍历 environment.yml 中的包，保持顺序
    for (pkg_name, pkg_version) in env_deps_list:
        low_name = pkg_name.lower()
        if low_name in installed_pkgs:
            current_ver = installed_pkgs[low_name]  # 实际安装的版本
            installed_list.append((pkg_name, pkg_version, current_ver))
        else:
            not_installed_list.append((pkg_name, pkg_version))

    print("========== 已安装的库 ==========")
    for pkg_name, yml_version, current_ver in installed_list:
        if yml_version:
            cmp_result = compare_versions(current_ver, yml_version)
        else:
            # environment.yml 未指定版本时，不比较
            cmp_result = None

        # 构造输出行
        if yml_version:
            line = f"[OK]  {pkg_name} 已安装版本({current_ver}) ( environment.yml中版本: {yml_version} )"
        else:
            line = f"[OK]  {pkg_name} 已安装版本({current_ver}) ( environment.yml未指定版本 )"

        # 根据比较结果给整行上色
        if cmp_result == ">":  # 当前版本更高 -> 绿色
            print(f"{GREEN}{line}{RESET}")
        elif cmp_result == "<":  # 当前版本更低 -> 红色
            print(f"{RED}{line}{RESET}")
        else:
            # 相等 或 无法比较 -> 默认颜色
            print(line)

    print("\n========== 未安装的库 ==========")
    for pkg_name, yml_version in not_installed_list:
        if yml_version:
            print(f"[NO]  {pkg_name} ( environment.yml中版本: {yml_version} )")
        else:
            print(f"[NO]  {pkg_name} ( environment.yml未指定版本 )")


if __name__ == "__main__":
    main()
