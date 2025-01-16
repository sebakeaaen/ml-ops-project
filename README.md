# mlops

# DTU ML Ops project. Pistachio classification.
Our project is based on a dataset of pistachio images, aiming to classify them into 2 classes (types): Kirmizi or Siirt. Link to the dataset: https://www.kaggle.com/datasets/muratkokludataset/pistachio-image-dataset. The goal is to set up a complete Machine Learning pipeline covering development, deployment, maintenance and operations of a pipeline for detecing the pistachio type based on the given images.

We will be using the PyTorch image models library (timm) for our modelling, and expect to use the PyTorch Lightning framework to remove the need for boilerplate. 

The dataset consists of 2148 .jpeg images:
  - Normal → 1232 images
  - Siirt → 916 images

There are also textual features, but we will not be using these.

Since our dataset of choice contains relatively limited amounts of data, we will try an use some pretrained models, perhaps a resnet variant as a first stab.




## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
