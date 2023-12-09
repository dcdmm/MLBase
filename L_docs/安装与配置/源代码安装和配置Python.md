1. 安装源码编译依赖的库
   ```shell
   sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev
   
   sudo apt install wget
   ```

2. 下载 Python 源码
   ```shell
   wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tgz
   ```

3. 安装包解压
   ```shell
   tar -xzf Python-3.8.12.tgz
   ```

4. 编译安装
   ```shell
   cd Python-3.8.12
   
   ./configure --prefix=/usr/local/python3.8
   
   make && make install
   ```

5. 设置软链接(最好不要覆盖系统中原有的python,python3,python3.?,pip,pip3,pip3.?,若要覆盖,最好先备份)