1. 下载anconda

    ```shell
    wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
    ```

2. 安装anconda(yes即可)

    ```shell
    bash Anaconda3-2021.05-Linux-x86_64.sh 
    ```

3. vim ~/.bashrc

    ```shell
    # 文件结尾添加下面语句
    export PATH=$PATH:/root/anaconda3/bin
    ```

4. souce ~/.bashrc