import os

# 当前工作目录(调用文件决定)
cwd_path = os.getcwd()
# print->C:\Users\dcdmm\Music\GitHubProjects\MLNote\G_Tool\other\操作系统相关os模块\diff-getcwd-__file__\dir
print(cwd_path)

# 当前文件路径(当前文件决定)
file_path = os.path.abspath(__file__)
# print->C:\Users\dcdmm\Music\GitHubProjects\MLNote\G_Tool\other\操作系统相关os模块\diff-getcwd-__file__\dir\inner_diff.py
print(file_path)
