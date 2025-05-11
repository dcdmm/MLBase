```shell
# Create a virtual environment.
# By default, creates a virtual environment named .venv in the working directory. An alternative path may be provided positionally.
uv venv 

# The path to the virtual environment to create.
# Relative paths are resolved relative to the working directory.
uv venv `obj_name`

# --python
# The Python interpreter to use for the virtual environment.
uv venv --python `3.12`
uv venv --python `3.12.7`
uv venv --python `>=3.12`
```

```shell
# List the available Python installations.
# By default, installed Python versions and the downloads for latest available patch version of each supported Python major version are shown.
uv python list

# List all Python versions, including old patch versions.
uv python list --all-versions

# Download and install Python versions.
# Python versions are installed into the uv Python directory, which can be retrieved with uv python dir.
uv python install `3.13`
uv python install `3.13.1`

# Show the uv Python installation directory.
uv python dir
```

```shell
# List, in tabular format, packages installed in an environment
uv pip list

# Install all listed packages.
# The order of the packages is used to determine priority during resolution.
uv pip install `numpy pandas`  

# -r:
# Install all packages listed in the given requirements.txt files.
uv pip install -r `requirements.txt`

# Uninstall all listed packages
uv pip uninstall `numpy pandas`

# Show information about one or more installed packages
uv pip show `numpy pandas`

# Display the dependency tree for an environment
uv pip tree
# example
#joblib v1.4.2
#pandas v2.2.3
#├── numpy v2.2.5
#├── python-dateutil v2.9.0.post0
#│   └── six v1.17.0
#├── pytz v2025.2
#└── tzdata v2025.2
#regex v2024.7.24

# List, in requirements format, packages installed in an environment
uv pip freeze > `requirements.txt`

# Compile a requirements.in file to a requirements.txt file
# -o: 
# Write the compiled requirements to the given requirements.txt or pylock.toml file.
uv pip compile `pyproject.toml`  -o `requirements.txt`  # 与`uv export`命令类似

# Sync an environment with a requirements.txt file.
# When syncing an environment, any packages not listed in the requirements.txt file will be removed. To retain extraneous packages, use uv pip install instead.
uv pip sync `requirements.txt`
```

```shell
# Create a new project.
uv init `project_name`

# --python:
# The Python interpreter to use to determine the minimum supported Python version.
uv init --python 3.12 `project_name`
uv init --python ">=3.12.1" `project_name`

# Add dependencies to the project.
# Dependencies are added to the project's pyproject.toml file.
uv add `numpy`

# Remove dependencies from the project.
# Dependencies are removed from the project's pyproject.toml file.
uv remove `numpy`

# Update the project's lockfile.
uv lock  # pyproject.toml信息同步到uv.lock

# Update the project's environment.
# Syncing ensures that all project dependencies are installed and up-to-date with the lockfile.
uv sync  # uv.lock信息同步到虚拟环境

# Display the project's dependency tree
uv tree
#├── numpy v2.2.5
#└── pandas v2.2.3
#    ├── numpy v2.2.5
#    ├── python-dateutil v2.9.0.post0
#    │   └── six v1.17.0
#    ├── pytz v2025.2
#    └── tzdata v2025.2

# -d:Maximum display depth of the dependency tree [default: 255]
uv tree -d 1  
# ├── baostock v0.8.9
# ├── jupyter v1.1.1
# └── pandas v2.2.3
uv tree -d 2
# ├── baostock v0.8.9
# │   └── pandas v2.2.3
# ├── jupyter v1.1.1
# │   ├── ipykernel v6.29.5
# │   ├── ipywidgets v8.1.7
# │   ├── jupyter-console v6.6.3
# │   ├── jupyterlab v4.4.2
# │   ├── nbconvert v7.16.6
# │   └── notebook v7.4.2
# └── pandas v2.2.3 (*)

# Export the project's lockfile to an alternate format.
# -o: Write the exported requirements to the given file
uv export -o `requirements.txt`  # 与`uv pip compile`命令类似
```

```shell
# Clear the cache, removing all entries or those linked to specific packages
uv cache clean
```
