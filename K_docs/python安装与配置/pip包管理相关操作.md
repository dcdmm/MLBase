* Ubuntu系统默认python环境(/usr/bin/)安装pip:
    ```shell
    apt install python3-pip
    ```

* 当前pip版本信息
    ```shell
    pip -V
    ```

* 生成requirements.txt文件
    ```shell
    # pip freeze only saves the packages that are installed with pip install in your environment.
    # pip freeze saves all packages in the environment including those that you don't use in your current project (if you don't have virtualenv)
    pip list --format=freeze > requirements.txt  # 基于当前环境;将requirements 导出到指定文件中,默认home目录
    
    # pip install pipreqs
    # Usage:pipreqs [options] [<path>]
    # <path>:The path to the directory containing the application files for which a requirements file should be generated (defaults to the current working directory)
    # Generate requirements.txt file for any project based on imports
    pipreqs ./ --encoding=utf8 # 基于当前项目,requirements.txt位置:当前项目第一层目录下
    ```

* requirements.txt文件批量安装
    ```shell
    pip install -r requirements.txt
    ``` 

* 指定镜像源安装
    ```shell
    # 阿里源
    pip install paddlenlp -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
    
    # 清华源
    pip install paddlenlp -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
    ```