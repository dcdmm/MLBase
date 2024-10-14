import os

# os.environ behaves like a python dictionary, so all the common dictionary operations like get and set can be performed.
# We can also modify os.environ but any changes will be effective only for the current process where it was assigned and it will not change the value permanently.

print(os.environ)  # 所有环境变量

print(os.environ.get('JAVA_HOME'))  # 获取环境变量

print(os.environ.get('obj_environ'))  # None
print(os.environ.get('obj_environ', "not_found"))  # not_found

os.environ['obj_environ'] = 'hello python'  # 增加环境变量
print(os.environ.get('obj_environ'))

os.environ['obj_environ'] = "hello rust"  # 修改环境变量
print(os.environ.get('obj_environ'))

del os.environ['obj_environ']  # 删除环境变量
print(os.environ.get('obj_environ'))
