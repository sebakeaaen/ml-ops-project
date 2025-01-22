# Base image
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

RUN dvc init --no-scm
COPY .dvc/config .dvc/config
COPY *.dvc /
RUN dvc config core.no_scm true
RUN dvc pull

ENTRYPOINT ["python", "-u", "src/mlops/train.py"]
