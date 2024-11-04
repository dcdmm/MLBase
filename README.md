### 目录

* A_PythonBasis
    * Other
    * 内置文件读写
    * 函数式编程
    * 基础语法
    * 常用函数[类]大全
    * 常见数据类型
    * 异常处理
    * 类型注解
    * 运算符
    * 面向对象编程
* B_Numpy
* C_Pandas
    * 1.基本数据类型介绍
    * 2.基本方法介绍
    * 3.数据清洗
    * 4.文件读写
    * 5.联合和合并数据集
    * 6.高级方法
    * 7.数据聚合与分组操作
    * 8.时间序列数据处理
* D_Plot======>>几个主流绘图库的使用
    * graphviz
    * matplotlib
        * other
        * 基础知识======>>>color/colorbar/marker/spine/title/xlabel/xticks/xaxis/grid/text/legend/子图......的使用介绍
        * 统计图像======>>>3D图/填图/散点图/柱状图/直方图/等高线图/线图/饼图的......的使用介绍
    * other======>其他作图功能
    * plotly
    * seaborn
* E_PyTorch
    * other
    * 创建张量
    * 基本操作
    * 形状操作
    * 索引切片
    * 连接,拆分操作
    * 高阶操作及深度学习相关理论
        * torh.nn神经网络模型======>>>容器/线性层/卷积层/池化层/激活函数层/循环层/丢弃层/初始化.....的相关原理及使用介绍
        * torch.optim优化算法======>>>常见优化器(SGD/Adagrad/Adam......)及学习率调整策略
        * torch.utils.data======>>>数据处理
        * torch.utils.tensorboard======>>>可视化
        * torchvision======>>>计算机视觉
* E_Tensorflow
    * other
    * 创建张量
    * 基本操作
    * 形状操作
    * 索引切片
    * 连接,拆分操作
    * 高阶操作及深度学习相关理论
* F_Tool======>>>其他实用库的相关介绍与使用
    * other
    * 伪随机数random模块
    * 命令行解析argparse模块
    * 复杂网络结构networkx模块
    * 容器collections模块
    * 性能优化与对象存储joblib模块
    * 操作系统相关os模块
    * 数据验证pydantic模块
    * 日志系统logging模块
    * 时间操作相关模块
    * 机器学习sklearn模块
    * 正则表达式re模块
    * 深拷贝浅拷贝copy模块
    * 符号运算sympy模块
    * 自然语言处理HuggingFace工具集
    * 自然语言处理SentenceTransformers模块
    * 迭代器itertools模块
* G_BasicTheory
    * 机器学习======>>>多分类/数据集划分/数据预处理/模型评估与选择/超参数搜索/特征选择......的相关原理及使用介绍
    * 概率论与数理统计======>>>基本概念(均值,协方差,标准差......)/概率分布(正态,t,F,卡方,泊松......)/统计分析/估计算法的相关原理及使用介绍
    * 线性代数======>>>基本概念(正交矩阵,梯度矩阵,向量内积,向量外积,逆矩阵......)/矩阵分析的相关原理及使用介绍
    * 高等数学======>>>基本概念(积分,求导,复数运算,三角函数......)/数据分析(凸优化基础,插值法,最优化方法......)的相关原理及使用介绍
* H_Model======>>介绍主流模型的理论及其实现,并以实例说明
    * classic_ML_model
    * deep_learning_base
    * other
    * task_NER_RelationExtraction
    * task_SentenceEmbeddings
    * 图神经网络
    * 掩码语言模型MLM
* I_MyCode
    * encapsulation======>功能封装
    * tricks======>>技巧
    * 小知识大集合
* J_demo=====>一些实战小demo
* K_docs======>相关介绍文档与面经
* Other=====>数据,图谱
* tests


### 个人习惯规约

1. 数学关键字(.md),如:<font color='red' size=4>定理:</font>;<font color='red' size=4>定义:</font>;<font color='red' size=4>证明:</font>;<font color='red' size=4>分析:</font>$ ...... $
2. 算法伪代码标识(.md),如:**for**  $ i=1,2,\dots, n $ **do**
3. 代码or代码块标识(多行注释中),如:`import`;`%run`;`print("hello java)`;```print("hello python")```
4. 文件/文件夹命名:编号\_名称\_实现方式\_所用数据\_备注\_待完成?\_重要?_环境(难以区分时使用OOO进行分隔,+表示任意字符,xxx表示任何内容)
    * a\_example1\_sklearn\_mnist\_多分类问题\_ing\_erOOOpytorch_env.ipynb
    * Dropout+d.ipynb(包括Dropout1d、Dropout2d、Dropout3d的学习)
    * datasets_Dataset_f2_from_xxx.ipynb

### 数学符号系统

$ x $: 标量      
$ \mathbf{x}$:向量or序列(不能加粗的字母使用"\boldsymbol"修饰,如$\boldsymbol{\mu}$)
$ X $: 矩阵or随机变量or数据集   
$ \mathbf{I}_n$:n行n列单位矩阵
$ \mathcal{X} $: 样本空间或状态空间,也可以用来表示概率分布,如$\mathcal{D}$     
$ \mathbb{I}(*) $:  指示函数,在$*$为真/假时分别取值为1/0   
$ \mathrm{sign}(*) $:  符号函数,在<0,=0,>0时分别取值为-1,0,1  
$ E_{* \sim \mathcal{D}} [f( * )] $: 函数$f$对$ * $在分布$\mathcal{D}$下的数学期望;明确意义时可省略$\mathcal{D}$[和,或]$ * $