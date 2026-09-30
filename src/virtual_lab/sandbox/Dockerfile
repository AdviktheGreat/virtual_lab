# syntax=docker/dockerfile:1

# The software Biomni's agent writes code against, as an image code from a meeting can run in.
#
# The environment files in biomni_env/ are copied unchanged from snap-stanford/Biomni (Apache
# License 2.0; see NOTICE), and each stage installs them the way Biomni's biomni_env/setup.sh
# does. There are three stages, since the whole environment takes hours to build and tens of
# gigabytes to hold:
#
#   base  Python 3.11 with Biomni's analysis libraries (environment.yml)
#   bio   base, plus the bioinformatics tools and libraries (bio_env.yml)
#   full  bio, plus R with Biomni's CRAN and Bioconductor packages, the command-line tools in
#         cli_tools_config.json, and the software added in Biomni 0.0.8
#
# Build one with virtual_lab.build_sandbox_image, or by hand from this directory:
#
#   docker build --target base --tag virtual-lab-sandbox:base .
#
# Biomni publishes its environment for x86-64 Linux only (pystan has no ARM wheels, and several
# command-line tools are x86-64 binaries), so build bio and full with --platform linux/amd64.

ARG MINIFORGE_VERSION=25.3.1-0

FROM condaforge/miniforge3:${MINIFORGE_VERSION} AS base

ENV DEBIAN_FRONTEND=noninteractive

# What Biomni's scripts assume a machine already has: a compiler for pip packages without
# wheels, git and git-lfs for packages installed from repositories, and the download and JSON
# tools install_cli_tools.sh uses
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git git-lfs jq unzip wget \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system

WORKDIR /opt/biomni_env

COPY biomni_env/environment.yml ./
RUN conda env create -n biomni_e1 -f environment.yml \
    && conda clean -afy \
    && rm -rf /root/.cache

# The environment comes first on the path, as though activated, so that "python3" and
# "Rscript" in a run are its own. Code runs as the host's user, who has no home in the image,
# and the root filesystem is read-only at run time, so every cache goes under /tmp.
ENV PATH=/opt/conda/envs/biomni_e1/bin:/opt/conda/bin:$PATH \
    CONDA_DEFAULT_ENV=biomni_e1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache \
    NUMBA_CACHE_DIR=/tmp/numba \
    PIP_NO_CACHE_DIR=1

RUN python3 -c "import numpy, pandas, scipy, sklearn, statsmodels, matplotlib, seaborn, networkx"


FROM base AS bio

COPY biomni_env/bio_env.yml ./
RUN conda env update -n biomni_e1 -f bio_env.yml \
    && conda clean -afy \
    && rm -rf /tmp/.cache

RUN python3 -c "import scanpy, Bio, rdkit, pysam, gseapy, lifelines" \
    && samtools --version \
    && bwa 2>&1 | grep -q Version \
    && blastn -version


FROM bio AS full

COPY biomni_env/r_packages.yml biomni_env/install_r_packages.R ./
RUN conda env update -n biomni_e1 -f r_packages.yml \
    && conda clean -afy \
    && Rscript install_r_packages.R \
    && rm -rf /tmp/Rtmp* /tmp/.cache

COPY biomni_env/install_cli_tools.sh biomni_env/cli_tools_config.json ./
RUN BIOMNI_AUTO_INSTALL=1 BIOMNI_TOOLS_DIR=/opt/biomni_tools SHELL=/bin/bash bash install_cli_tools.sh \
    && rm -rf /tmp/.cache /tmp/.bashrc*

ENV PATH=/opt/biomni_tools/bin:$PATH

# Biomni 0.0.8's additions (new_software_v008.sh), less the lines that need sudo or a shell
# profile: git-lfs is installed above, and uv's tools go where every user can run them
COPY biomni_env/new_software_v008.sh ./
# uv must not fetch an interpreter of its own: it would keep it under HOME, which is /tmp and
# so hidden by the scratch space at run time, leaving the tools that use it broken
ENV UV_TOOL_DIR=/opt/uv/tools \
    UV_TOOL_BIN_DIR=/usr/local/bin \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never
RUN pip install --no-cache-dir transformers sentencepiece langchain-google-genai langchain_ollama mcp \
    && pip install --no-cache-dir lazyslide \
    && pip install --no-cache-dir "git+https://github.com/YosefLab/popV.git@refs/pull/100/head" \
    && pip install --no-cache-dir uv pybiomart fair-esm \
    && uv pip install --system --python "$(command -v python3)" transcriptformer \
    && pip install --no-cache-dir "zarr>=2.0,<3.0" \
    && uv tool install --python "$(command -v python3)" arc-state \
    && pip install --no-cache-dir nnunet nibabel nilearn mi-googlesearch-python \
    && pip install --no-cache-dir "git+https://github.com/pylabrobot/pylabrobot.git" \
    && conda install -y -n biomni_e1 -c conda-forge weasyprint \
    && conda clean -afy \
    && rm -rf /tmp/.cache

# install_cli_tools.sh exits successfully even when a tool fails to install, so each binary it
# should have put in place is checked for here
RUN Rscript -e 'library(DESeq2); library(clusterProfiler)' \
    && jq -r '.tools[].binary_path' cli_tools_config.json | while read -r binary; do \
        test -x "/opt/biomni_tools/bin/$(basename "$binary")" \
            || { echo "$binary was not installed" >&2; exit 1; }; \
    done \
    && plink2 --version \
    && iqtree2 --version \
    && state --help > /dev/null
