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
dependencies = ["torch", "sub0"]

[tool.uv.sources]
torch = { index = "pt_index" }  # ensure that torch is always installed from the pt_index index
# The workspace = true key-value pair in the tool.uv.sources table indicates the sub0 dependency should be provided by the workspace, rather than fetched from PyPI or another registry.
sub0 = { workspace = true } 

# Defining an index
# By default, uv includes the Python Package Index (PyPI) as the "default" index, i.e., the index used when a package is not found on any other index. 
[[tool.uv.index]]
name = "pt_index"
url = "https://download.pytorch.org/whl/cpu"

[tool.uv.workspace]
members = ["sub0", "sub1"]
```