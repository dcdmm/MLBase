### 常见损失函数数据类型要求

L1Loss: Input/Target 至少一个为float 

MSELoss: Input/Target 至少一个为float

NLLLoss: Input:float  Target:int    可指定ignore_index

CrossEntropyLoss: Input:float  Target:int   可指定ignore_index 

BCEloss: Input:[0 ~ 1]float  Target:[0 ~ 1] float

BCEWithLogitsLoss: Input:float  Target:[0 ~ 1] float