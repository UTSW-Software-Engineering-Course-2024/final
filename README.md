# final
this is the repository for the final exam

## I. Set up 
Here we use `python=3.9.18` because `tensorflow=2.10` don't support `python 3.11`

### 1. set up virtual environment

- If you don't have `virtualenv` package, first install it.
```
python -m pip install --user virtualenv
python -m virtualenv --help
```
- Install virtual python environment
```
virtualenv <Your_Env_Name> --python=python3.9
```
- activate the python environment
```
source <Path_To_Env>/bin/activate
```
- install the python package required

```
pip install -r requirement.txt
```

- load openslide module(in biohpc)

```
module load openslide/3.4.0
```

- check if it runs correctly

```
import tensorflow as tf
import openslide
```


