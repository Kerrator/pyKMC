# Installation 

## Python Environment 

It is recommended to use a python environment, with a version of python >= 3.9, dedicated to the use of pyKMC to avoid python packages conflict.
Using the virtual environment tool `venv` to create a new python environment : 
```bash 
python3 -m venv /path_to_environment/pykmc_env 
``` 
and to activate the newly created environment : 
```bash 
source /path_to_environment/pykmc_env/bin/activate
```
Then, to install pyKMC with all his depedencies : 
```bash 
cd path_to/pyKMC
pip install -e .
```



