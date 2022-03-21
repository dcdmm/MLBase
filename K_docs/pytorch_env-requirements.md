### pytorch_env虚拟环境安装

#### 机器学习/数据分析

* xgboost: pip install xgboost(linux下先执行pip install --upgrade pip)
* lightgbm: pip install lightgbm
* catboost: pip install catboost
* imblear: [https://imbalanced-learn.org/stable/install.html](https://imbalanced-learn.org/stable/install.html) 查看
* bayesian-optimization: pip install bayesian-optimization
* optuna: pip install optuna
* seaborn: pip install seaborn
* heamy: pip install -U heamy

#### 深度学习

* PyTorch: [https://pytorch.org/](https://pytorch.org/) 查看
* pytorch-tabnet: pip install pytorch-tabnet(依赖pytorch)
* torchtext: conda install -c pytorch torchtext
* torchsummary: pip install torchsummary
* tensorflow(CPU版本): pip install tensorflow
* pip install tensorflow-addons
* pydot(keras.utils.plot_model函数依赖): pip install pydot
* pydot-ns(keras.utils.plot_model函数依赖): pip install pydot-ng
* FastText:
    * Linux: [https://github.com/facebookresearch/fastText](https://github.com/facebookresearch/fastText) 查看
    * Windows:
        * https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext 下载安装*.whl
        * cd *.whl所在有目录
        * pip install *.whl

#### 自然语言处理

* transformers(Hugging Face): pip install transformers
* datasets(Hugging Face): pip install datasets
* gensim: [https://radimrehurek.com/gensim/](https://radimrehurek.com/gensim/) 查看
* jieba: pip install jieba
* nltk: pip install nltk
* spacy(CPU版本):
    * pip install -U pip setuptools wheel
    * pip install -U spacy
    * python -m spacy download zh_core_web_sm
    * python -m spacy download en_core_web_sm

#### 其他

* wget: pip install wget
* PyQt5(交互方式绘图依赖): pip install PyQt5
* graphviz:
    * Windows:pip install graphviz(需要安装Graphviz并配置系统环境变量Path +: Graphviz\bin)
* jupyter notebook目录功能安装及其配置
    * pip install jupyter_contrib_nbextensions
    * pip install jupyter_nbextensions_configurator
    * jupyter contrib nbextension install --user
* PyMySQL: pip install PyMySQL
* autopep8(vscode代码格式化): pip install --upgrade autopep8 