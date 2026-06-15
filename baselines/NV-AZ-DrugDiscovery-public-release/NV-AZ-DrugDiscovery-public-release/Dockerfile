# syntax=docker/dockerfile:1

FROM mambaorg/micromamba:1.5.8

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY environment.yml /tmp/environment.yml

RUN micromamba env create -n edm -f /tmp/environment.yml \
    && micromamba clean --all --yes

SHELL ["micromamba", "run", "-n", "edm", "/bin/bash", "-c"]

COPY . /workspace

ENV PATH=/opt/conda/envs/edm/bin:$PATH
ENV CONDA_DEFAULT_ENV=edm

CMD ["bash"]
