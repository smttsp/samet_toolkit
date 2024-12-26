# samet_toolkit

This repo is a toolkit for my personal and professional projects. 
It contains the following tools (and will expand).

1. LLM
    - Embedding generator 
    - Text File Parser: .pdf, .txt, .docx for now
    -  
2. Database operations
    - Supabase
    - Pinecone
    - Firebase
3. 

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

### Creating Python Environment

The installation on Windows can be done with conda. 

The first step is to download a miniconda installer from the following link:

https://docs.conda.io/en/latest/miniconda.html

Once it is installed and conda is available in the command prompt, you can create a 
new environment with the following command:

```bash 
conda create -n new_environment python=3.11.5
```

Activate the environment with the following command:

```bash
conda activate new_environment
```

### Installing Dependencies

Install the dependencies with the following command:

```bash
pip install poetry
poetry install --no-root
```
