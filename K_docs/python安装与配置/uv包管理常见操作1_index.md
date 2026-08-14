```toml
# Defining an index
# By default, uv includes the Python Package Index (PyPI) as the "default" index, i.e., the index used when a package is not found on any other index. 
[[tool.uv.index]]
# Optional name for the index.
name = "default"
url = "https://mirrors.aliyun.com/pypi/simple/" 
default = true
```


```toml
[project]
dependencies = ["torch"]

[tool.uv.sources]
# ensure that torch is always installed from the pt_index index
torch = { index = "pt_index" }  

[[tool.uv.index]]
name = "pt_index"
url = "https://download.pytorch.org/whl/cpu"
```