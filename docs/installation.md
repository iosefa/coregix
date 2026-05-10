# Installation

Coregix is published as a Python package and can be installed with `pip`.

## Requirements

Coregix requires Python 3.9 or newer. Runtime dependencies are installed with the package:

- `rasterio`
- `numpy`
- `affine`
- `itk-elastix`

For most users, installing from PyPI is the right starting point.

## Install From PyPI

```bash
pip install coregix
```

This installs the Python API and the `vhr-align-image-pair` command-line entrypoint.

## Verify The Install

Check the command-line entrypoint:

```bash
vhr-align-image-pair --help
```

You can also run the CLI module directly:

```bash
python -m coregix.cli.align_image_pair --help
```

## Developer Install

For local development, clone the repository and create the conda environment:

```bash
git clone https://github.com/iosefa/coregix.git
cd coregix
conda env create -f environment.yml
conda activate coregix
```

The environment installs Coregix in editable mode with the documentation extra enabled:

```bash
pip install -e ".[docs]"
```

You do not need to run that command separately when using `environment.yml`.

If you already have a compatible environment and only want the editable package install:

```bash
pip install -e .
```

To include documentation dependencies in an existing environment:

```bash
pip install -e ".[docs]"
```

## Preview The Documentation

Run the local documentation server from the repository root:

```bash
mkdocs serve
```

Open `http://127.0.0.1:8000` in a browser. MkDocs watches the documentation files and refreshes the site after edits.

To build the static site:

```bash
mkdocs build
```

The generated HTML is written to `site/`.
