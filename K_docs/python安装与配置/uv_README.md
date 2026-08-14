When running a command that mutates an environment such as uv pip sync or uv pip install, uv will search for a virtual environment in the following order:

* An activated virtual environment based on the VIRTUAL_ENV environment variable.
* An activated Conda environment based on the CONDA_PREFIX environment variable.
* A virtual environment at .venv in the current directory, or in the nearest parent directory.