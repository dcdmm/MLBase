### 个人习惯规约

1. 数学关键字(.md),如:<font color='red' size=4>定理:</font>;<font color='red' size=4>定义:</font>;<font color='red' size=4>
   证明:</font>;<font color='red' size=4>分析:</font>$ ...... $
2. 算法伪代码标识(.md),如:**for**  $ i=1,2,\dots, n $ **do**
3. 代码or代码块标识(注释中),如:`import`;`%run`;`print("hello java)`
4. 文件/文件夹命名,编号\_名称\_实现方式\_所用数据\_备注\_待完成?\_重要?_环境(难以区分时使用OOO进行分隔),如:a\_example1\_sklearn\_mnist\_多分类问题\_ing\_erOOOpytorch_env

### 符号系统

$ x $: 标量      
$ \mathbf{x}$:向量or序列(不能加粗的字母使用"\boldsymbol"修饰,如$\boldsymbol{\mu}$)
$ X $: 矩阵or随机变量or数据集   
$ \mathbf{I}_n$:n行n列单位矩阵 
$ \mathcal{X} $: 样本空间或状态空间,也可以用来表示概率分布,如$\mathcal{D}$     
$ \mathbb{I}(*) $:  指示函数,在$*$为真/假时分别取值为1/0   
$ \mathrm{sign}(*) $:  符号函数,在<0,=0,>0时分别取值为-1,0,1  
$ E_{* \sim \mathcal{D}} [f( * )] $: 函数$f$对$ * $在分布$\mathcal{D}$下的数学期望;明确意义时可省略$\mathcal{D}$[和,或]$ * $

### 模块使用习惯

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
    * 均可
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
    * 除(定义语句):
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

### 主要参考(排名不分先后)

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
* 跟李沐学AI(B站)
* 2021春机器学习课程(B站)(李宏毅)
* Natural Language Processing with Attention Models(吴恩达)
* python/numpy/pandas/matplotlib/sklearn/pytorch/tensorflow等官网
* ......