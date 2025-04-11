import yaml
import argparse
import sys


class MyParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(MyParser, self).__init__(*args, **kwargs)

    def parse_args_and_config(self):
        args = sys.argv[1:]
        # 如果命令行参数中包含--config
        if "--config" in args:
            config_index = args.index("--config")
            args.pop(config_index)  # 弹出--config
            config_path = args.pop(config_index)  # 弹出并获得--config后的配置文件
            with open(config_path, encoding='utf-8') as yaml_file:
                config = yaml.safe_load(yaml_file)
                if config:
                    # 通过配置文件信息更新命令行参数
                    for action in self._actions:
                        if action.dest in config:
                            action.default = config[action.dest]
        # 更新sys.argv
        sys.argv = [sys.argv[0]] + args


def my_argument():
    parser = MyParser(
        description='模型训练参数'
    )
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--lr_scheduler_type', type=str, default="linear")
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.parse_args_and_config()
    return parser


if __name__ == '__main__':
    para = my_argument()
    print(vars(para.parse_args()))
