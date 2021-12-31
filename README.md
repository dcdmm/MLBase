#### 个人习惯规约

1. 数学公式关键字(md),如:<font color='red' size=4>定理:</font>;<font color='red' size=4>定义:</font>;<font color='red' size=4>
   证明:</font> $ ...... $
2. 算法伪代码标识(md),如:**for**  $ i=1,2,\dots, n $ **do**
3. 代码or代码块标识(注释中),如:`import`;`%run`;`print("hello java)`
4. 文件/文件夹命名,编号_名称_实现方式_所用数据_备注_待完成?(难以区分使用OOO进行区分),如:a_OOO__name__OOO_sklearn_mnist_多分类_ing
5. notebook笔或帮助文件命名,最后加"_help"(预览效果排序:jupyter notebook>vscode>pycharm;后缀为:"_notebook"表示需使用jupyter notebok预览)
6. 重要度标识
    1. 非常重要:★★★★★
    2. 很重要:★★★★
    3. 重要:★★★
    4. 相对重要:★★

#### 符号系统

$ x $: 标量      
$ \mathbf{x}$:向量or序列(某些字符不能加粗,使用"\boldsymbol",如$\boldsymbol{\mu}$)
$ X $: 矩阵or随机变量or数据集   
$ \mathbf{I}_n$:n行n列单位矩阵 $ \mathcal{X} $: 样本空间或状态空间,也可以用来表示概率分布,如$\mathcal{D}$     
$ \mathbb{I}(*) $:  指示函数,在$*$为真/假时分别取值为1/0   
$ \mathrm{sign}(*) $:  符号函数,在<0,=0,>0时分别取值为-1,0,1  
$ E_{* \sim \mathcal{D}} [f( * )] $: 函数$f( * )$对$ * $在分布$\mathcal{D}$下的数学期望;明确意义时可省略$\mathcal{D}$[和,或]$ * $

#### 模块使用习惯

1. numpy模块
    * np.\*
        * np.reshaep
        * np.where
        * np.isinf
        * np.random.shuffle
        * ....
    * arr.\*
        * .astype
        * .flatten
    * 均使用
        * reshape
        * ravel
2. pandas模块
    * 均$*.*$
        * .replace
        * .fillna
        * .merge
        * .sum
        * .concat
        * ...
    * 除:
        * pd.DataFrame
        * pd.Series
        * pd.date_range
        * pd,to_datetime
        * pd.period_range
        * pd.read_excel
        * pd.read_csv
        * ...
        * pd.set_option
3. pytorch模块
    1. torch.\*
        1. torch.autograd
        2. torch.nn
        3. torch.nn.functional
        4. torch.cuda
        5. torch.optim
        6. ...
    2. tor.\*
        1. .to
        2. .dim
        3. .size
        4. .tolist
        5. .numpy
        6. .item
        7. .backward
        8. .retain_grad
        9. .in-place操作
        10. .expand
        11. .repeat
    3. 均使用
        1. reshape
4. tensorflow
    1. tf.\*
        1. torch.constant
        2. torch.Variable
        3. torch.reshape
        4. torch.data
        5. torch.feature_columns
        6. torch.random
        7. torch.keras
        8. ...
    2. tor.\*
        1. .numpy
        2. .assign
        3. .assign_add
        4. .assign_sub

#### 额外安装(初始安装为Anconda环境)

1. 核心安装模块
    * xgboost: pip install xgboost(linux下先执行pip install --upgrade pip)
    * lightgbm: pip install lightgbm
    * catboost: pip install catboost
    * PyTorch: [https://pytorch.org/](https://pytorch.org/) 查看
    * pytorch-tabnet: pip install pytorch-tabnet(依赖pytorch)
    * torchtext: conda install -c pytorch torchtext
    * torchsummary: pip install torchsummary
    * tensorflow: pip install tensorflow-gpu
    * pydot: pip install pydot(keras.utils.plot_model依赖)
    * pydot-ns: pip install pydot-ng(keras.utils.plot_model依赖)
    * gensim: [https://radimrehurek.com/gensim/](https://radimrehurek.com/gensim/)查看(其他模块冲突,暂未解决,虚拟环境按照)
    * jieba: pip install jieba
    * spacy: [https://spacy.io/usage](https://spacy.io/usage) 查看(其他模块冲突,暂未解决,虚拟环境按照)
    * imblear: [https://imbalanced-learn.org/stable/install.html](https://imbalanced-learn.org/stable/install.html) 查看
    * bayesian-optimization: pip install bayesian-optimization
    * optuna: pip install optuna
    * graphviz: pip install graphviz(需要安装Graphviz并配置系统环境变量Path +: Graphviz\bin)
    * wget: pip install wget
2. jupyter notebook目录功能安装及其配置
    1. pip install jupyter_contrib_nbextensions
    2. pip install jupyter_nbextensions_configurator
    3. jupyter contrib nbextension install --user

#### 主要参考(排名不分先后)

* <<统计学习方法>>(李航)
* <<机器学习>>(周志华)
* <<神经网络与深度学习>>(邱锡鹏)
* <<利用Python进行数据分析>>(Wes Mckinney)
* <<深度学习入门-基于Python的理论与实现>>(斋藤康毅)
* <<Pattern Recognition and Machine Learning>>(Christopher M. Bishop)
* <<数学分析>>(华东师范大学)
* <<高等数理统计>>(茆诗松)
* <<统计推断>>(George Casella, Roger L. Berger)
* <<矩阵分析与应用>>(张贤达)
* 白板推导(B站)
* python/numpy/pandas/matplotlib/seaborn/sklearn/pytorch等官网
* ......