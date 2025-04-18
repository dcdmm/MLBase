* 清理未使用的包和缓存
    ```shell
    conda clean --all
    ```

* 导出当前环境配置到文件
    ```shell
    conda env export > 文件名.yml
    ```

* 从文件创建conda虚拟环境环境
    ```shell
    conda env create -f 文件名.yml -n 自定义环境名
    ```