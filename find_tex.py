import os
import glob

def find_tex_files(root_dir):
    tex_files = []
    for dir_path, _, file_names in os.walk(root_dir):
        for file_name in file_names:
            if file_name.endswith('.tex'):
                tex_files.append(os.path.join(dir_path, file_name))
    return tex_files

# 指定根目录
root_directory = './tex_before'

# 查找所有.tex文件
tex_files = find_tex_files(root_directory)

tex_files = sorted(tex_files)
# 打印找到的.tex文件路径
for file in tex_files:
    print(f'"{file}",')