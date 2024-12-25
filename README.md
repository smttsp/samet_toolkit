# samet_tamplate

This repo is for template for my projects.


## Installation - Mac

### Prerequisite: `pyenv`

`pyenv` simplifies Python version management, enabling you to seamlessly switch between 
Python versions for different project requirements.



https://github.com/pyenv/pyenv-installer

On macOS you can use [brew](https://brew.sh), but you may need to grab the `--HEAD` version for the latest:

```bash
brew install pyenv --HEAD
```

or

```bash
curl https://pyenv.run | bash
```

And then you should check the local `.python-version` file or `.envrc` and install the correct version which will be the basis for the local virtual environment. If the `.python-version` exists you can run:

```bash
pyenv install
```

This will show a message like this if you already have the right version, and you can just respond with `N` (No) to cancel the re-install:

```bash
pyenv: ~/.pyenv/versions/3.8.6 already exists
continue with installation? (y/N) N
```

### Prerequisite: `direnv`

`direnv` streamlines environment variable management, allowing you to isolate 
project-specific configurations and dependencies within your development environment.

https://direnv.net/docs/installation.html

```bash
curl -sfL https://direnv.net/install.sh | bash
```

### Developer Setup

If you are a new developer to this package and need to develop, test, or build -- please run the following to create a developer-ready local Virtual Environment:

```bash
direnv allow
python --version
pip install --upgrade pip
pip install poetry
poetry install
```


## Installation - Windows

The installation on Windows can be done with conda. 

1. The first step is to download a miniconda installer from the following link:

https://docs.conda.io/en/latest/miniconda.html

2. Once it is installed and conda is available in the command prompt, you can create a new environment with the following command:

```bash 
conda create -n sttr python=3.11.5
```

3. Activate the environment with the following command:

```bash
conda activate sttr
```


4. Install the dependencies with the following command:

```bash
pip install poetry
poetry install --no-root
```


## Running the Application

To run the application, you need to run the following command:

```bash
python main.py
```

which will run GUI for the application. You can upload the image which will automatically 
trigger the backend and do the processing. The resulting BOM will then be displayed on
the screen.
