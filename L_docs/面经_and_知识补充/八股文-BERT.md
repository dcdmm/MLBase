#### BERT模型八股文
* Bert中为什么要在开头加个[CLS]?
   * 为了拿到更好的句子语义表示,更好意味着这个表征能够照顾大局,不对任何一个位置过分偏心
* Bert中有哪些地方用到了mask?
  1. 预训练阶段的mask有什么用
     * 构造完形填空的预训练任务
  2. attention中的mask有什么用?
     * 处理掉padding部分带来的无效信息 
  3. decoder中的mask有什么用?
     * 防止语言模型利用了leak未来信息
* Bert为什么要使用warmup的学习率trick?
  * 为了让开始阶段的数据分布更好,更容易训练,防止过拟合到少数几个batch上
* Bert如何处理一词多义?
  * 利用self-attention中不同词词和词之间的交互
* Bert中的transformer和原生的transformer有什么区别?
  * 原生用的是周期函数对相对位置进行编码,Bert换成了position embedding
* Bert是如何处理传统方法难以搞定的溢出词表词(oov)的语义学习的?
  * Bert使用subword,把词拆碎,常见的typo或者语言特殊表达,都能有一部分照顾到
  