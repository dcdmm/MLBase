### 核心包安装指南

#### 机器学习/数据分析

* xgboost: pip install xgboost(linux下先执行pip install --upgrade pip)
* lightgbm: pip install lightgbm
* catboost: pip install catboost
* imblear: [https://imbalanced-learn.org/stable/install.html](https://imbalanced-learn.org/stable/install.html) 查看
* bayesian-optimization: pip install bayesian-optimization
* optuna: pip install optuna
* seaborn: pip install seaborn
* plotly: pip install plotly
* heamy: pip install -U heamy
* networkx: pip install networkx
* iterative-stratification(多标签分层k折): pip install iterative-stratification

#### 深度学习

* PyTorch: [https://pytorch.org/](https://pytorch.org/) 查看
* pytorch-tabnet: pip install pytorch-tabnet(依赖pytorch)
* pytorch-crf:[https://pytorch-crf.readthedocs.io/en/stable/](https://pytorch-crf.readthedocs.io/en/stable) 查看
* torchinfo: conda install -c conda-forge torchinfo
* pytorch-lightning: pip install pytorch-lightning
* tensorflow(CPU版本): pip install tensorflow
    * pydot(keras.utils.plot_model函数依赖): pip install pydot
    * pydot-ns(keras.utils.plot_model函数依赖): pip install pydot-ng
* tensorflow-addons: pip install tensorflow-addons
* FastText:
    * Linux: [https://github.com/facebookresearch/fastText](https://github.com/facebookresearch/fastText) 查看
    * Windows:
        * https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext 下载安装*.whl
        * cd *.whl所在有目录
        * pip install *.whl
* wandb: pip install wandb

#### 自然语言处理

* transformers(Hugging Face): pip install transformers
* datasets(Hugging Face): pip install datasets
* sentence_transformers: pip install -U sentence-transformers
* nltk: pip install nltk
* pyahocorasick: pip install pyahocorasick

#### 其他

* pipreqs(requirements.txt文件生成): pip install pipreqs
* wget: pip install wget
* joblib: pip install joblib
* colorama(彩色打印): pip install colorama
* SymPy: pip install sympy
* jsonlines: pip install jsonlines
* graphviz:
    * Windows:pip install graphviz(需要安装Graphviz并配置系统环境变量Path +: Graphviz\bin)
* conda默认包依赖项
    * XlsxWriter(pandas保存excel文件依赖): pip install XlsxWriter
    * openpyxl(pandas读取excel文件依赖): pip install openpyxl
    * tabulate(pandas DataFrame保存markdown格式依赖): pip install tabulate
    * PyQt5(交互方式绘图依赖): pip install PyQt5
    * autopep8(vscode代码格式化依赖): pip install --upgrade autopep8