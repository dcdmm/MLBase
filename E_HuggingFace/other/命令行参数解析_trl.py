from dataclasses import dataclass
from trl import GRPOConfig
from trl import TrlParser  # 继承自transformers.HfArgumentParser


@dataclass
class d1:
    d10: float = 5e-5
    d11: int = 32


def main():
    parser = TrlParser((d1, GRPOConfig))

    # 使用方式一(不含--config命令行参数 + yaml配置文件): python 命令行参数解析_trl.py
    # parser.parse_args_and_config()
    # all_args = parser.parse_args()
    # print(vars(all_args), end='\n\n')
    # print(type(vars(all_args)), end='\n\n')  # print->dict
    # d1_args, grpo_args = parser.parse_args_into_dataclasses()
    # print(d1_args, end='\n\n')
    # print(type(d1_args), end='\n\n')  # print-><class '__main__.d1'>
    # print(grpo_args, end='\n\n')
    # print(type(grpo_args))  # print-><class 'transformers.training_args.TrainingArguments'>

    # 使用方式二(含--config命令行参数 + yaml配置文件): python 命令行参数解析_trl.py --config config.yaml
    d1_args1, grpo_args = parser.parse_args_and_config()
    print(d1_args1)  # print->d1(d10=5e-05, d11=100)
    print(grpo_args.max_steps)  # print->100000
    print(grpo_args.seed)  # print->2025


if __name__ == "__main__":
    main()
