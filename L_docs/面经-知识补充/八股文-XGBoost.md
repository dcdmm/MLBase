#### XGBoost八股文

* XGBoost与GBDT的区别与联系
  1. 在使用CART作为基分类器时,XGBoost显式地加入了正则项来控制模型的复杂度,有利于防止过拟合,从而提高模型的泛化能力
     * $ \Omega(f)=\bar{\gamma} T+\frac{1}{2} \bar{\lambda}\|w\|^2 $;$T$:叶子节点个数,$\|w\|^2$:叶子权重的L2范数
  2. GBDT在模型训练时只使用了损失函数的一阶导数信息,XGBoost对代价函数进行二阶泰勒展开(可以更为精准的逼近真实的损失函数),可以同时使用一阶和二阶导数
  3. 传统的GBDT采用CART作为基分类器,XGBoost支持多种类型的基分类器,比如线性分类器
  4. XGBoost借鉴了随机森林的做法,支持列抽样(即每次的输入特征不是全部特征),可以降低过拟合
  5. 传统的GBDT没有设计对缺失值进行处理,XGBoost能够自动学习出缺失值的处理策略
     1. 在特征k上寻找最佳分割点,不会对该列特征缺失的样本进行遍历,而只对该列特征值为非缺失样本上对应的特征值进行遍历,通过这个技巧来减少了为稀疏离散特征寻找分割点的时间开销
     2. 在逻辑实现上,为了保证完备性,会将该特征值缺失的样本分别分配到左叶子结点和右叶子结点,两种情形都计算一遍后,选择分裂后增益最大的那个方向(左分支或是右分支),作为预测时特征值缺失样本的默认分支方向
     3. 如果在训练中没有缺失值而在预测中出现缺失,那么会自动将缺失值的划分方向放到右子结点
* XGBoost都有哪些防止过拟合的设计
  1. 目标函数加入正则项:叶子节点个数+叶子节点权重的L2正则化
  2. 列抽样:训练的时候只用一部分特征
  3. 子采样:每轮计算可以不使用全部样本,使算法更加保守
  4. shrinkage:可以叫学习率或步长,每次迭代,增加新的模型,在前面乘上一个小于1的系数,降低优化的速度,每次走一小步逐步逼近最优模型比每次走一大步逼近更加容易避免过拟合现象
 * XGBoost如何评价特征的重要性(For tree model Importance type can be defined as:)
   * 'weight': the number of times a feature is used to split the data across all trees.
   * 'gain': the average gain across all splits the feature is used in.
   * 'cover': the average coverage across all splits the feature is used in.
   * 'total_gain': the total gain across all splits the feature is used in.
   * 'total_cover': the total coverage across all splits the feature is used in.