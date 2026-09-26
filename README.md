# Virtual Lab

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/virtual-lab)](https://badge.fury.io/py/virtual-lab)
[![PyPI version](https://badge.fury.io/py/virtual-lab.svg)](https://badge.fury.io/py/virtual-lab)
[![Downloads](https://pepy.tech/badge/virtual-lab)](https://pepy.tech/project/virtual-lab)
[![license](https://img.shields.io/github/license/zou-group/virtual-lab.svg)](https://github.com/zou-group/virtual-lab/blob/main/LICENSE.txt)

![Virtual Lab](https://github.com/zou-group/virtual-lab/raw/main/images/virtual_lab_architecture.png)

The **Virtual Lab** is an AI-human collaboration for science research. In the Virtual Lab, a human researcher works with a team of large language model (LLM) **agents** to perform scientific research. Interaction between the human researcher and the LLM agents occurs via a series of **team meetings**, where all the LLM agents discuss a scientific agenda posed by the human researcher, and **individual meetings**, where the human researcher interacts with a single LLM agent to solve a particular scientific task.

Please see our paper [The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies](https://www.nature.com/articles/s41586-025-09442-9) for more details on the Virtual Lab and an application to nanobody design for SARS-CoV-2.

If you use the Virtual Lab, please cite our work as follows:

Swanson, K., Wu, W., Bulaong, N.L. et al. The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies. *Nature* (2025). https://doi.org/10.1038/s41586-025-09442-9


## Virtual Lab for nanobody design

As a real-world demonstration, we applied the Virtual Lab to design nanobodies for one of the latest variants of SARS-CoV-2 (see [nanobody_design](https://github.com/zou-group/virtual-lab/tree/main/nanobody_design)). The Virtual Lab built a computational pipeline consisting of [ESM](https://www.science.org/doi/10.1126/science.ade2574), [AlphaFold-Multimer](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2), and [Rosetta](https://rosettacommons.org/software/) and used it to design 92 nanobodies that were experimentally validated.

Please see the notebook [nanobody_design/run_nanobody_design.ipynb](https://github.com/zou-group/virtual-lab/blob/main/nanobody_design/run_nanobody_design.ipynb) for an example of how to use the Virtual Lab to create agents and run team and individual meetings.


## Installation

The Virtual Lab requires Python 3.12 or later. It can be installed using pip or by cloning the repo and installing the required packages. Installation should only take a couple of minutes.

Optionally, first create a conda environment.

```bash
conda create -y -n virtual_lab python=3.12
conda activate virtual_lab
```

Python 3.12 is recommended rather than a newer version because reproducing the nanobody design study relies on `nanobody_design/requirements_nanobody_design_frozen.txt`, which pins `torch==2.4.1`. That release has no wheels beyond Python 3.12. If you only need the `virtual_lab` package itself, any version from 3.12 onwards works.

The Virtual Lab can be installed via pip.

```bash
pip install virtual-lab
```

To install the latest version of the Virtual Lab locally, clone the repo and then install the package.

```bash
git clone https://github.com/zou-group/virtual_lab.git
cd virtual_lab
pip install -e .
```


## OpenAI API Key

The Virtual Lab currently uses GPT-5.2 from OpenAI by default. Save your OpenAI API key as the environment variable `OPENAI_API_KEY`. For example, add `export OPENAI_API_KEY=<your_key>` to your `.bashrc` or `.bash_profile`.


## Running the code that agents write

Agents can write code, and the Virtual Lab can run it. Because that code is written by a model and read by nobody before it runs, it is executed in a container rather than on your machine. Install [Docker](https://docs.docker.com/get-started/get-docker/) and make sure it is running.

```python
from pathlib import Path
from virtual_lab import DockerExecutor, run_files, save_artifacts

paths = save_artifacts(save_dir=Path("results"), save_name="discussion", artifacts=code)
results = run_files(
    directory=paths[0].parent, files=code.files, executor=DockerExecutor()
)
```

The container has no network access, no view of your filesystem beyond the meeting's own output directory, a read-only root filesystem, and hard limits on memory, processes, and wall clock time. It is not given your environment, so the code cannot read your API key. Scripts that genuinely need to reach the internet require `DockerExecutor(allow_network=True)`.

The first run pulls the sandbox image, which takes a minute. Files the code writes into its working directory appear on your machine; everything else it does is discarded with the container.

If you do not have Docker, `LocalExecutor` runs code directly on your machine instead. It is not a sandbox: code run through it can read and write any file you can. It applies a timeout and withholds your API key, and that is the extent of it.


## Fixing code that fails

`run_with_repair` writes a meeting's code, runs it, and hands the traceback back to the agent that wrote it, asking for a correction. It stops as soon as the code runs, when the attempts are used up, or when an attempt reproduces the previous error.

```python
from virtual_lab import DockerExecutor, run_with_repair, save_execution_record

outcome = run_with_repair(
    artifacts=code,
    author=machine_learning_specialist,
    save_dir=Path("results"),
    save_name="discussion",
    executor=DockerExecutor(),
)
save_execution_record(
    save_dir=Path("results"),
    save_name="discussion",
    outcome=outcome,
    author=machine_learning_specialist,
    executor=DockerExecutor(),
)
```

The failure goes to the author rather than to the critic because the author knows what the code was meant to do, while the critic's job is scientific judgement rather than debugging. Each attempt past the first costs another round of code generation, so `max_attempts` defaults to 3. The record written to `executions/<save_name>.json` lists every attempt and what it produced, alongside the total token usage and cost of the repairs, making the multiplier visible rather than buried.

The result can then be reviewed as evidence rather than as a claim, by passing it into the next meeting as context:

```python
run_meeting(
    meeting_type="individual",
    agenda="Review the results of running the analysis.",
    save_dir=Path("results"),
    team_member=machine_learning_specialist,
    contexts=(outcome.report(),),
)
```

`outcome.report()` states plainly whether the code worked, whether it had to be corrected first, and whether it was abandoned. A critic that is not told the code failed three times will review the code as though it had worked.

## Querying scientific databases

Tools that look something up go through `virtual_lab.web`, which decides where a request may be sent. An agent chooses the arguments to a tool, which means a model's output determines part of every URL, so the destination cannot be left to the tool:

- Requests are allowed only to the scientific databases named in `ALLOWED_HOSTS`, over HTTPS. Anything else raises `DisallowedHostError`, including a host that merely ends in an allowed one, a plaintext address, a loopback or link-local address, and a non-HTTP scheme.
- Redirects are followed manually and the destination is checked at every hop. A client left to follow redirects itself checks the address it was given, not the address it ends up talking to, which turns one allowed host into a request to anywhere.
- Identifiers are percent-encoded into a fixed template by `build_url`, so an identifier containing `/`, `?`, or `#` becomes part of the path rather than reshaping the request.
- Responses are capped at `MAX_RESPONSE_BYTES` whether or not the service declares a length, requests to one host are spaced by `MIN_SECONDS_BETWEEN_REQUESTS`, temporary failures are retried with backoff that honours `Retry-After`, and repeated questions are answered from an in-memory cache.

```python
from virtual_lab import request_json

data = request_json("https://rest.uniprot.org/uniprotkb/P01308.json")
```

Adding a database means adding its host to `ALLOWED_HOSTS`. This is a deliberate edit rather than a parameter, so that widening what agents can reach is a visible change to the library.

The tests for this layer run offline against a faked transport, since redirects, rate limits, and oversized responses cannot be requested from a real service on demand. The handful that do query the real databases are skipped unless `VIRTUAL_LAB_LIVE_TESTS=1` is set.
