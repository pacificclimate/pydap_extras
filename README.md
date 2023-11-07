# pydap_extras

Consolidates all PCIC Pydap handlers and responses. Uses the mainline Pydap for Python 3.

## Development and testing

It is best practice to install Python projects in a virtual environment. We use [Poetry](https://python-poetry.org/) to manage Python package dependencies and virtual environments. All instructions in this section assume that Poetry is installed on the host system. 

### Installation

This package uses GDAL version 3.0.4. That makes it very tricky to install correctly. The following rigmarole appears to the only way to get a successful installation. A brief explanation follows:

1. GDAL 3.0.4 depends on `numpy`, which is apparently undeclared but is flagged as a warning during installation and causes the installation to fail. The version must be `numpy<=1.21`. Pre-installing `numpy` solves this problem.

2. GDAL 3.0.4 requires something called `use_2to3`. Modern versions of `setuptools` do not support it; only versions `setuptools<58` do. See, for example,

   -   https://github.com/nextgis/pygdal/issues/67
   -   https://github.com/pypa/setuptools/issues/2781
   -   https://github.com/OSGeo/gdal/issues/7541

   We must therefore explicitly install `setuptools<58`.

3. Poetry somehow still stumbles over installing `gdal==3.0.4`. However, it can be installed via Poetry into the virtualenv by using `pip3`. 

Hence:

```bash 
# Pre-install numpy 
poetry run pip3 install numpy==1.21
# Install required version of setuptools for installing gdal==3.0.4
poetry run pip3 install "setuptools<58"
# Install gdal using pip3
poetry run pip3 install gdal==3.0.4
# Install rest of project
poetry install
```

### Running unit tests

```bash  
poetry run pytest
```

## App

The app will run on port 8001.

```bash  
pip install -r test_requirements.txt
python pydap_extras/app.py [filepath]
```
