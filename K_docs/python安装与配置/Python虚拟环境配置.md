1. 创建虚拟环境
    ```shell
   # 创建一个新的虚拟环境,指定python版本为3.10.12
    conda create -n your_env_name python=3.10.12
   
   # 通过克隆创建一个与已有虚拟环境(可以是base环境,也可以是其他已存在的虚拟环)相同的新环境
   conda create -n your_env_name --clone base 
    ```

2. 更新bashrc中的环境变量
    ```shell
    # Linux
    conda init bash && source 用户目录/.bashrc
    
    # Windows
    conda init bash
    ```

3. 进入虚拟环境
    ```shell
    conda activate your_env_name
    ```

### 其他常见命令

```shell
# 查看当前存在的虚拟环境
conda env list

# 关闭虚拟环境(该虚拟环境中)
conda deactivate

# 删除某个虚拟环境
conda remove -n your_env_name --all
```

