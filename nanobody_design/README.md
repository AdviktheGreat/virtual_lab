# Nanobody Design

This README contains the commands needed to apply the Virtual Lab to nanobody design for one of the latest variants of SARS-CoV-2.


## Installation

After following the general `virtual_lab` installation [instructions](https://github.com/zou-group/virtual-lab/blob/main/README.md), run the following to install nanobody-specific requirements:

```bash
conda activate virtual_lab
cd /path/to/virtual_lab
pip install -e .[nanobody-design]
```

If there are any version incompatibility issues, please run `pip install -r nanobody_design/requirements_nanobody_design_frozen.txt`. These are the exact versions used for the paper, and they require Python 3.12 (`torch==2.4.1` has no wheels for later versions).

Additionally, create a separate virtual environment for running AlphaFold-Multimer via LocalColabFold by following these instructions: https://github.com/YoshitakaMo/localcolabfold (or by running `install_localcolabfold.sh`). Make sure the following versions are installed:

```
alphafold-colabfold==2.3.6
colabfold==1.5.5
```


## LLM project design

Run the `run_nanobody_design.ipynb` notebook to have LLM agents create a nanobody design workflow. Each meeting with the LLM agents should only take a few minutes. The outputs of these notebooks are in the `nanobody_design/discussions` directory. The relevant ESM, AlphaFold-Multimer, and Rosetta scripts have been copied from those discussions to the `nanobody_design/scripts` folder. Additionally, the sequences for the relevant nanobodies selected by the LLM agents, along with SARS-CoV-2 spike RBD sequences, are in the `nanobody_design/sequences` directory.


## Nanobody design

Starting with four nanobodies designed to bind to the original Wuhan strain of SARS-CoV-2---Ty1, H11-D4, Nb21, and VHH-72---improve them by iteratively adding mutations and scoring them using ESM, AlphaFold-Multimer, and Rosetta for four rounds.

See commands in `nanobody_design/workflow.md`.
