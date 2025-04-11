from dataclasses import dataclass
from transformers import HfArgumentParser  # 继承自argparse.ArgumentParser
from transformers import TrainingArguments  # 数据类


@dataclass
class d1:
    d10: float = 5e-5
    d11: int = 32


@dataclass
class d2:
    d20: int = 3
    d21: str = "./output"


def main():
    parser = HfArgumentParser((d1, d2, TrainingArguments))
    all_args = parser.parse_args()
    print(vars(all_args), end='\n\n')
    print(type(vars(all_args)), end='\n\n')  # print->dict

    d1_args, d2_args, ta_args = parser.parse_args_into_dataclasses()
    print(d1_args, end='\n\n')
    print(type(d1_args), end='\n\n')  # print-><class '__main__.d1'>
    print(ta_args, end='\n\n')
    print(type(ta_args))  # print-><class 'transformers.training_args.TrainingArguments'>


# python 命令行参数解析.py
# python 命令行参数解析.py --output_dir "xx"
if __name__ == "__main__":
    main()