# pydap_extras

Consolidates all PCIC Pydap handlers and responses. Uses the mainline Pydap for Python 3.

## Development and testing

It is best practice to install Python projects in a virtual environment. We use [Poetry](https://python-poetry.org/) to manage Python package dependencies and virtual environments. All instructions in this section assume that Poetry is installed on the host system. 

### Installation

This package uses GDAL version 3.0.4. It is tricky to install correctly. The following rigmarole, specifically with pre-installations and a special version of `setuptools`, appears to the only way to get a successful installation. A brief explanation follows:

1. GDAL 3.0.4 depends on `numpy`, which is apparently undeclared but is flagged by `gdal` as a warning if it is not installed already, and causes the installation to fail. The version must be `numpy<=1.21`. Pre-installing `numpy` solves this problem.

2. GDAL 3.0.4 requires something called `use_2to3`. Modern versions of `setuptools` do not support it; only versions `setuptools<58` do. See, for example,

   -   https://github.com/nextgis/pygdal/issues/67
   -   https://github.com/pypa/setuptools/issues/2781
   -   https://github.com/OSGeo/gdal/issues/7541

   We must therefore explicitly install `setuptools<58` before we install `gdal`.

3. Poetry somehow still stumbles over installing `gdal==3.0.4` using its own tooling. However, it can be installed via Poetry into the virtualenv by using `pip3`. This circumvents whatever Poetry does. 
4. Once the above steps have been taken, the installation can be completed using the normal `poetry install`.
5. Note that dependencies have been organized into groups to make this as simple as possible. If and when later versions of GDAL are specified, this organization and the installation steps are likely to need to be updated. (Possibly, it will become simpler.) 

Hence:

```bash 
# Pre-install initial packages (numpy, setuptools) 
poetry install --only initial
# Install gdal using pip3 into the Poetry virtualenv
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
