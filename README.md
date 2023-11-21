# pydap_extras

Consolidates all PCIC Pydap handlers and responses. Uses the mainline Pydap for Python 3.

## Development and testing

It is best practice to install Python projects in a virtual environment. We use [Poetry](https://python-poetry.org/) to manage Python package dependencies and virtual environments. All instructions in this section assume that Poetry is installed on the host system. 

### Installation

This package uses GDAL version 3.0.4. It is tricky to install correctly. The following rigmarole, specifically with pre-installations and a special version of `setuptools`, appears to the only way to get a successful installation. A brief explanation follows:

1. GDAL 3.0.4 requires something called `use_2to3`. Modern versions of `setuptools` do not support it; only versions `setuptools<58` do. See, for example,

   -   https://github.com/nextgis/pygdal/issues/67
   -   https://github.com/pypa/setuptools/issues/2781
   -   https://github.com/OSGeo/gdal/issues/7541

   We must therefore explicitly install `setuptools<58` before we install `gdal`.

2. GDAL 3.0.4 cannot be installed successfully by later versions of `pip`. Version 22.3.1 does work. We must ensure it is installed before installing `gdal`.

3. GDAL 3.0.4 depends on `numpy`. This is apparently not declared as a dependency but _is_ flagged by `gdal` as a warning if it is not already installed, and causes the installation to fail. The version must be `numpy<=1.21`. Pre-installing `numpy` solves this problem.

4. Poetry somehow still stumbles over installing `gdal==3.0.4` using its own tooling. However, `gdal` can be installed via Poetry into the virtualenv by using the appropriate version of `pip` (see previous item). This circumvents whatever Poetry does. 

5. Once the above steps have been taken, the installation can be completed using the normal `poetry install`.

6. Note that dependencies have been organized into groups to make this as simple as possible. If and when later versions of GDAL are specified, this organization and the installation steps are likely to need to be updated. (Possibly, it will become simpler.) 

Hence:

```bash 
# Pre-install initial packages (pip, setuptools, numpy) 
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
poetry run python pydap_extras/app.py [filepath]
```

## Releasing

1. Modify `tool.poetry.version` in `pyproject.toml`.
1. Summarize release changes in `NEWS.md`
1. Commit these changes, then tag the release
   ```bash
   git add pyproject.toml NEWS.md
   git commit -m"Bump to version X.Y.Z"
   git tag -a -m"X.Y.Z" X.Y.Z
   git push --follow-tags
   ```
1. Our GitHub Action `pypi-publish.yml` will build and release the package
   on our PyPI server.
