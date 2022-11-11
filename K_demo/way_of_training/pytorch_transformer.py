import torch
import torch.utils.data as Data
from datasets import load_dataset, load_from_disk
from transformers import BertTokenizer
from transformers import BertModel
from sklearn.metrics import accuracy_score
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from transformers import Trainer, TrainingArguments
import numpy as np
import random
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(lineno)d - %(message)s ',
                    filename='pytorch_transformer.log')


def set_seed(seed):
    """PyTorch随机数种子设置大全"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # CPU上设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 当前GPU上设置随机种子
        # A bool that, if True, causes cuDNN to only use deterministic convolution algorithms.
        torch.backends.cudnn.deterministic = True
        # torch.cuda.manual_seed_all(seed) # 所有GPU上设置随机种子


seed = 2022
set_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(msg="device:" + str(device))


class Dataset(Data.Dataset):
    """定义数据集"""

    def __init__(self, split):
        self.split = split
        self.dataset = load_dataset(path='seamew/ChnSentiCorp', split=self.split)

    # 必须实现__len__魔法方法
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        """定义索引方式"""
        text = self.dataset[i]['text']
        if self.split == 'compare_test':
            return text,  # 测试数据集不含标签
        else:
            label = self.dataset[i]['label']
            return text, label


dataset_train = Dataset('train')
dataset_validation = Dataset('validation')
dataset_test = Dataset('compare_test')

for text, label in dataset_train:
    # 调用__getitem__方法
    logging.info(msg="dataset_train text First for cycle:" + str(text))
    logging.info(msg="dataset_train label First for cycle:" + str(label))
    break

for text in dataset_test:
    # 调用__getitem__方法
    logging.info(msg="dataset_test text First for cycle:" + str(text))  # 元组 
    break

model_ckpt = "bert-base-chinese"
token = BertTokenizer.from_pretrained(model_ckpt)
logging.info(msg="token.model_input_names:" + str(token.model_input_names))
pretrained = BertModel.from_pretrained(model_ckpt)
logging.info(msg="pretrained.num_parameters():" + str(pretrained.num_parameters()))


# 冻结网络层参数(不进行梯度更新)
for param in pretrained.parameters():
    param.requires_grad = False


def get_collate_fn(tokenizer, max_len=512):
    """返回collate_fun函数(通过闭包函数引入形参)"""

    def collate_fn(data):
        sents = [i[0] for i in data]

        # 批量编码句子
        text_token = tokenizer(text=sents,
                               truncation=True,
                               padding='max_length',
                               max_length=max_len,
                               return_token_type_ids=True,
                               return_attention_mask=True,
                               return_tensors='pt')

        input_ids = text_token['input_ids']
        attention_mask = text_token['attention_mask']
        token_type_ids = text_token['token_type_ids']
        # 返回值必须为字典(键与模型forward方法形参对应)
        result = {'input_ids': input_ids,  # ★★★★★对应模型forward方法input_ids参数
                  'attention_mask': attention_mask,  # ★★★★★对应模型forward方法attention_mask参数
                  "token_type_ids": token_type_ids}  # ★★★★对应模型forward方法token_type_ids参数

        if len(data[0]) == 1:
            return result  # 测试数据集不含标签
        else:
            labels = [i[1] for i in data]
            labels = torch.LongTensor(labels)
            result['labels'] = labels  # ★★★★对应模型forward方法labels参数
            return result

    return collate_fn


dataLoader_test = Data.DataLoader(dataset=dataset_test, batch_size=2, collate_fn=get_collate_fn(token, max_len=512))
for i in dataLoader_test:
    logging.info(msg="dataLoader_test First for cycle:" + str(i))
    break


class Model(torch.nn.Module):
    """下游训练任务模型"""

    def __init__(self, pretrained_model):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)  # 二分类任务
        self.pretrained = pretrained_model
        self.criterion = torch.nn.CrossEntropyLoss()  # 损失函数

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        out = self.pretrained(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)

        out = self.fc(out.pooler_output)
        out = out.softmax(dim=1)
        loss = None
        if labels is not None:  # 若包含标签
            loss = self.criterion(out, labels)

        # 训练与评估阶段
        # ★★★★★
        # 返回值为一个元组
        # 元组的第一个元素必须为该批次数据的损失值
        # 元组的第二个元素为该批次数据的预测值(可选)
        # * 验证数据集评估函数指标的计算
        # * predict方法预测结果(predictions)与评估结果(metrics)(结合输入labels)的计算
        if loss is not None:
            return (loss, out)
        # 预测阶段
        # ★★★★★
        # 返回值为模型的预测结果
        else:
            return out

model = Model(pretrained)
model = model.to(device)


def compute_metrics(pred):
    """验证数据集评估函数"""
    labels = pred.label_ids  # 对应自定义模型forward函数输入:labels
    preds = pred.predictions  # 对应自定义模型forward函数返回值的第二个元素
    preds_argmax = preds.argmax(-1)
    acc = accuracy_score(labels, preds_argmax)
    return {"accuracy": acc}  # return a dictionary string to metric value


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # 学习率预热(线性增加)
            return float(current_step) / float(max(1, num_warmup_steps))
        # 学习率线性衰减(最小为0)
        # num_training_steps后学习率恒为0
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda)


model_name = f"{model_ckpt}-finetuned-emotion"
batch_size = 64  # 批次大小
epochs = 5.0  # 训练轮数
steps_all = int(len(dataset_train) / batch_size) * epochs  # 总学习步数
optimizer = optim.AdamW(model.parameters(), lr=5e-4)  # 优化器
scheduler_lr = get_linear_schedule_with_warmup(optimizer, 50, 0.9 * steps_all)  # 学习率预热(必须为LambdaLR对象)


# 主要调节的超参数
training_args = TrainingArguments(
    # The output directory where the model predictions and checkpoints will be written.
    output_dir=model_name,
    seed=42,

    # Total number of training epochs to perform
    num_train_epochs=epochs,  # 默认:3.0
    # If set to a positive number, the total number of training steps to perform. Overrides num_train_epochs. I
    # max_steps=100,  # 默认:-1

    #  Maximum gradient norm (for gradient clipping).
    max_grad_norm=1.0,  # 默认:1.0

    # 对应pytorch DataLoader 参数batch_size
    # The batch size per GPU/TPU core/CPU for training.
    per_device_train_batch_size=batch_size,  # 默认:8
    # The batch size per GPU/TPU core/CPU for evaluation.
    # 对应pytorch DataLoader 参数batch_size
    per_device_eval_batch_size=batch_size,  # 默认:8
    # Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch size) or not.
    # 对应pytorch DataLoader 参数drop_last
    dataloader_drop_last=False,  # 默认:False

    # The evaluation strategy to adopt during training. Possible values are:
    # "no": No evaluation is done during training.
    # "steps": Evaluation is done (and logged) every eval_steps.
    # "epoch": Evaluation is done at the end of each epoch.
    evaluation_strategy="epoch",  # 默认:'no'
    # The logging strategy to adopt during training. Possible values are:
    # "no": No logging is done during training.
    # "epoch": Logging is done at the end of each epoch.
    # "steps": Logging is done every logging_steps.
    logging_strategy='epoch',  # 默认:'steps'
    # Number of update steps between two logs if logging_strategy="steps".
    # logging_steps=500,  # 默认:500
    # The checkpoint save strategy to adopt during training. Possible values are:
    # "no": No save is done during training.
    # "epoch": Save is done at the end of each epoch.
    # "steps": Save is done every save_steps.
    # Logger log level to use on the main process. Possible choices are the log levels as strings: ‘debug’, ‘info’, ‘warning’, ‘error’ and ‘critical’, plus a ‘passive’ level which doesn’t set anything and lets the application set the level.
    log_level='passive',  # 默认'passive'
    save_strategy='epoch',  # 默认:'steps'
    # Number of updates steps before two checkpoint saves if save_strategy="steps".
    # save_steps=500,  # 默认:500
    disable_tqdm=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_validation,
    data_collator=get_collate_fn(token, max_len=512),  # 对应pytorch DataLoader 参数collate_fn
    optimizers=(optimizer, scheduler_lr),  # 自定义优化器与学习率预热
    compute_metrics=compute_metrics,
    tokenizer=token)

trainer.train()  # 模型训练

