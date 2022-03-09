### 额外安装(初始安装为Anconda环境)

1. 核心安装模块
    * xgboost: pip install xgboost(linux下先执行pip install --upgrade pip)
    * lightgbm: pip install lightgbm
    * catboost: pip install catboost
    * PyTorch: [https://pytorch.org/](https://pytorch.org/) 查看
    * pytorch-tabnet: pip install pytorch-tabnet(依赖pytorch)
    * torchtext: conda install -c pytorch torchtext
    * torchsummary: pip install torchsummary
    * tensorflow: pip install tensorflow-gpu(GPU版本)
    * pydot: pip install pydot(keras.utils.plot_model函数依赖)
    * pydot-ns: pip install pydot-ng(keras.utils.plot_model函数依赖)
    * transformers(Hugging Face): pip install transformers
    * datasets(Hugging Face): pip install datasets
    * gensim: [https://radimrehurek.com/gensim/](https://radimrehurek.com/gensim/)查看
    * jieba: pip install jieba
    * nltk: pip install nltk
    * spacy: [https://spacy.io/usage](https://spacy.io/usage) 查看(其他模块冲突,暂未解决,虚拟环境:spacy_test)
    * pip install nltk
    * imblear: [https://imbalanced-learn.org/stable/install.html](https://imbalanced-learn.org/stable/install.html) 查看
    * bayesian-optimization: pip install bayesian-optimization
    * optuna: pip install optuna
    * graphviz: pip install graphviz(需要安装Graphviz并配置系统环境变量Path +: Graphviz\bin)
    * wget: pip install wget
2. jupyter notebook目录功能安装及其配置
    1. pip install jupyter_contrib_nbextensions
    2. pip install jupyter_nbextensions_configurator
    3. jupyter contrib nbextension install --user