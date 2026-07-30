```toml
# 设置默认package indexes(dependency resolution and package installation)
[[tool.uv.index]]
# Optional name for the index.
name = "default"
url = "https://mirrors.aliyun.com/pypi/simple/"  # 默认:https://pypi.org/simple
default = true
```


```toml
[project]
dependencies = ["torch"]

[tool.uv.sources]
torch = { index = "pt_index" }  # ensure that torch is always installed from the pt_index index

[[tool.uv.index]]
name = "pt_index"
url = "https://download.pytorch.org/whl/cpu"
```