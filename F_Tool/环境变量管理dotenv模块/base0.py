from dotenv import load_dotenv, find_dotenv, dotenv_values
import os

print(find_dotenv())  # 逐级向上搜索找到第一个是.env的文件

# Parse a .env file and then load all the variables found as environment variables.
load_dotenv(dotenv_path=None)  # 默认dot_env_path为None,表示使用find_dotenv()找到的第一个.env文件

print(os.getenv("email"))
print(os.getenv("port"))
print(os.getenv("host"))

# Parse a .env file and return its content as a dict.
config = dotenv_values("test.env")
print(config)
