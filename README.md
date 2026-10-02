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


## Models and API keys

The Virtual Lab uses GPT-5.2 from OpenAI by default, and any agent can use a model from another provider. Each model is reached through LangChain, and its provider is worked out from its name by the rules [Biomni](https://github.com/snap-stanford/Biomni) uses, so an agent created with `model="claude-sonnet-4-5"` is sent to Anthropic and one with `model="gemini-2.5-pro"` to Google. Agents in the same meeting can use different providers.

| Provider | Model names | Key or setting |
| --- | --- | --- |
| OpenAI | `gpt-*`, `o1*`, `o3*`, `o4*`, `ft:*` | `OPENAI_API_KEY` |
| Anthropic | `claude-*` | `ANTHROPIC_API_KEY` |
| Google Gemini | `gemini-*` | `GEMINI_API_KEY` |
| Groq | any name containing `groq` | `GROQ_API_KEY` |
| Azure OpenAI | `azure-<deployment>` | `OPENAI_API_KEY`, `OPENAI_ENDPOINT` |
| Ollama (local) | `llama*`, `qwen*`, `mistral*`, `gpt-oss*`, and names with a `/` | `pip install virtual-lab[ollama]` |
| Amazon Bedrock | `anthropic.claude-*`, `us.*`, and other Bedrock IDs | `AWS_REGION`, `pip install virtual-lab[bedrock]` |

Set `LLM_SOURCE` to one of those providers to override the rules for every model. A meeting sends a name the rules do not recognise, such as `chatgpt-4o-latest`, to OpenAI as it always has, and while `OPENAI_BASE_URL` points at a server of your own it sends names that look like open models there rather than to Ollama. For anything else, including a self-hosted server with an OpenAI-compatible API, build the model yourself and pass it in:

```python
from virtual_lab import get_llm, hold_meeting

served = get_llm("biomni-r0", source="Custom", base_url="http://localhost:30000/v1")
result = hold_meeting(..., chat_models={"biomni-r0": served})
```

`chat_models` also takes a single LangChain chat model for every agent, or a function from model name to chat model. The record of each meeting names the class that answered for each model and the server it was sent to.

OpenAI's API is told which agent wrote each earlier turn. Other providers have no field for this, so for them the other agents' turns are passed as user messages that begin with the speaker's name, rather than as the reader's own words.


## Holding a meeting and limiting what it spends

`hold_meeting` takes the same arguments as `run_meeting` and returns everything the meeting produced: the summary, the structured output if a schema was given, the usage, the provenance record, and where each was saved. `run_meeting` is kept for existing notebooks and returns only one of these.

```python
from virtual_lab import BudgetExceededError, hold_meeting

try:
    result = hold_meeting(
        meeting_type="individual",
        agenda="Propose three candidate epitopes.",
        save_dir=Path("results"),
        team_member=immunologist,
        num_rounds=2,
        max_cost=0.50,
        max_completion_tokens=4_000,
    )
    print(result.summary, result.cost)
except BudgetExceededError as error:
    print(f"Stopped at ${error.spent:.2f}; the turns so far are in results/partial/")
```

- `max_cost` is checked before every request, including each tool call and the structured output, so a meeting stops before the request it has no money left for. It can overrun by at most one request, which `max_completion_tokens` bounds.
- A limit is refused outright if any model in the meeting has no price in `virtual_lab.constants`, and a response that reports no usage stops a limited meeting with `CostUnknownError`. Either would otherwise count as free. Without a limit, `result.cost` is `None` rather than a wrong number.
- OpenAI, Anthropic, and Gemini models are priced. Gemini models with two rates are priced at the higher one, so their cost is an upper bound.
- Prices are matched exactly or to a dated snapshot of a listed model (`gpt-4o-2024-08-06`, `claude-sonnet-4-5-20250929`). A name that merely starts with a listed one, such as `gpt-5-pro` or `gpt-5.4`, is unpriced rather than billed at the cheaper model's rate.
- `on_usage` is called with the running usage after every response, for a caller keeping a total across meetings. An exception it raises stops the meeting like any other failure.
- `before_request` is called before every request, after `max_cost` is checked. An exception it raises stops the meeting before the request is sent, which is how several meetings sharing one budget are stopped together (see `Project` below).
- A failed or interrupted meeting, including one stopped by its limit or by Ctrl-C, saves its turns and its record, with the usage up to that point, under `save_dir/partial/`.
- Several models (GPT-5 and its mini, nano, and pro versions among them) refuse any temperature but their default. When the API refuses the temperature, the request is sent again without it, the model is remembered for the rest of the process, and the record lists it under `models_at_default_temperature`.


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

### Biomni's software and data in the sandbox

The default sandbox image is plain Python. For the software Biomni's agent works with, build a sandbox image from Biomni's own environment files, which ship with this package, and fetch Biomni's data lake to mount into it:

```python
from pathlib import Path
from virtual_lab import SANDBOX_PLATFORM, DockerExecutor, build_sandbox_image, download_data_lake

image = build_sandbox_image("bio")
lake = download_data_lake(Path("data/biomni_data/data_lake"))
executor = DockerExecutor(image=image, data_lake=lake.directory, platform=SANDBOX_PLATFORM)
```

| Stage | Adds | Size, roughly |
| --- | --- | --- |
| `base` | Python 3.11 with Biomni's analysis libraries (pandas, scikit-learn, statsmodels, transformers, and the rest of its `environment.yml`) | 3 GB |
| `bio` | Biomni's bioinformatics tools and libraries: samtools, BWA, BLAST, Bowtie 2, scanpy, RDKit, pysam, scvi-tools, and the rest of its `bio_env.yml` | 18 GB |
| `full` | R with Biomni's CRAN and Bioconductor packages, its command-line tools (PLINK 2, IQ-TREE, GCTA, and others), and the software added in Biomni 0.0.8 | tens of GB, hours to build |

Each stage includes the one before. An image is built once and reused, and is tagged with the package version and platform, so that an upgrade builds a new one. Biomni publishes its environment for x86-64 Linux only, so the images are built for `linux/amd64` even on an ARM machine, where Docker runs them under emulation (slower, but complete). Pass the same platform to `DockerExecutor`, as above, or Docker warns about the mismatch on every run's stderr. `build_sandbox_image("base", platform=None)` builds for the machine's own architecture instead, under a tag of its own; run it with `DockerExecutor(platform=None)`. The `bio` and `full` stages do not build on ARM.

The data lake is 76 files and about 11 GB. `download_data_lake` takes a list of names to fetch only some of them (see `DATA_LAKE` for what each holds), skips files already present, and never leaves a partial file under a real name, so an interrupted download is resumed by calling it again. The sandbox sees the directory read-only at `/biomni_data/data_lake`, which is also in the `BIOMNI_DATA_LAKE` environment variable. Keep it outside the directory meetings are saved to: a data lake that overlaps the directory code runs in is refused, since the code could write to it there.

If you do not have Docker, `LocalExecutor` runs code directly on your machine instead. It is not a sandbox: code run through it can read and write any file you can. It applies a timeout and withholds your API key, and that is the extent of it.

### A shared session for exploratory analysis

A finished script is run once. An analysis is explored instead: load the data, look at it, decide what to do next. A session keeps one Python interpreter running for the length of a meeting, the way Biomni's agent keeps its REPL, so a table one agent loads is still there for the next agent to use:

```python
from pathlib import Path
from virtual_lab import DockerSession, session_executor, session_tool

with DockerSession(Path("results/session"), executor=session_executor("bio")) as session:
    session.run("import pandas as pd\ndf = pd.read_csv('counts.csv')")
    print(session.run("df.shape").output)          # the value of a final expression is printed
    session.run("samtools --version", language="bash")
    tool = session_tool(session)                   # a run_code tool to give the meeting's agents
```

Each run returns a `CellResult` holding what the code printed (standard output and standard error together, in order), its status, the figures it drew, and the files it wrote. Code that fails or runs out of time is a result, not an exception. A matplotlib figure left open is saved under `plots/` and closed. R and bash code runs as a fresh `Rscript` or `bash` process in the same directory each time, as it does in Biomni.

By default a session runs in Biomni's `full` image with the network on, since Biomni's agent queries databases as it works. Pass `session_executor("bio")` or `session_executor("base")` for a stage you have built, `allow_network=False` to cut the network, and `data_lake=` to mount the data lake. Everything else about the container matches `DockerExecutor`. The limits are higher (8 GB of memory, 4 CPUs, 2 GB of scratch space) because a session holds a whole analysis in memory at once.

Code is stopped at its time limit (`timeout=`, 300 seconds by default) and the session keeps everything it had defined. Code that ignores the stop, and code that crashes the interpreter or runs out of memory, costs the session: the result's status is `"lost"`, and the next run starts a fresh interpreter, which its `start` number shows. Starting a session in an image Docker emulates takes about half a minute. `LocalSession` runs the same interpreter on your machine, with no isolation, like `LocalExecutor`. Pass `python=` to use an environment already installed there, such as Biomni's `biomni_e1`.

### Running code during a meeting

Give a meeting a session and its agents can run code in it as they talk, checking a claim against the data or computing a number instead of estimating it:

```python
from virtual_lab import DockerSession, hold_meeting, session_executor

with DockerSession(Path("results/session"), executor=session_executor("bio")) as session:
    result = hold_meeting(
        meeting_type="team",
        agenda="Which genes separate the two clusters in counts.csv?",
        save_dir=Path("results"),
        team_lead=principal_investigator,
        team_members=(bioinformatician, scientific_critic),
        num_rounds=2,
        session=session,
    )
```

Everyone in the meeting shares the one interpreter, so what one agent loads or computes, the next can use. By default each agent runs code by calling a `run_code` tool. With `code_actions="tags"` it instead writes the code between `<execute>` and `</execute>` in its reply and is answered between `<observation>` and `</observation>`, as Biomni's agent is. That works with any model, including ones served without tool calling. A block starting with `#!R` is R and one starting with `#!BASH` is a shell script.

An agent may run code, or call tools, up to 20 times in a turn with a session (5 without), after which it is asked to answer without them; `max_tool_iterations=` changes that. The session is started before the first request, so one that cannot start costs nothing, and it is left running afterwards, so the next meeting can pick up where this one left off. Every piece of code the meeting ran, with its full output, is saved to `sessions/<save_name>.json`, and each turn in the meeting's record lists the cells it ran and how each ended. The transcript shows the code beside its output.

### Biomni's tools, data, and know-how in a meeting

Biomni's package ships with this one, unchanged, so code in a session can import its tool functions (223 of them, from database queries to CRISPR screen design) the way Biomni's agent does: `from biomni.tool.database import query_uniprot`. The package is mounted read-only into the container. It is not imported on your machine: its descriptions are read as data, and only its know-how loader, which needs nothing beyond the standard library, is run there. Each meeting tells its agents what they can use: Biomni's tools, the data lake files that are mounted, the software in the image, and Biomni's know-how documents, which are given in full.

Biomni lists all of its software to its agent, whose environment has all of it. A session may run in the `bio` or `base` stage, which has only some, so before a meeting lists anything it checks, in the session, which of Biomni's software is installed and which of its tool modules import. It lists only those, and its record names what was left out and why: in the `bio` stage, for instance, R and its packages, and the genomics and bioimaging tools, whose modules need libraries only the `full` stage has. The check takes a few seconds, is not part of the session's history, and runs the imports in a process of its own, so it leaves the session as it found it. If it cannot be done, everything is listed, with a warning.

Listing everything adds about 40,000 tokens to every request, so by default the agent who closes the meeting is first asked which resources the agenda needs, as Biomni's agent asks before each task, and only those are listed. That costs one request of about 12,000 tokens, which counts towards `max_cost` and the meeting's usage. An answer that cannot be read lists everything, with a warning.

```python
from virtual_lab import Resources, available_resources

hold_meeting(..., session=session)                     # the lead picks what the agenda needs
hold_meeting(..., session=session, resources="all")    # list everything
hold_meeting(..., session=session, resources="none")   # list nothing

everything = available_resources(session)
chosen = Resources(tools=tuple(t for t in everything.tools if t["module"] == "biomni.tool.database"))
hold_meeting(..., session=session, resources=chosen)   # list exactly these
```

Tools and software are listed only when the session can import Biomni's tools (`biomni_tools=False` in `session_executor` or `LocalSession` turns this off), and data lake files only when the data lake is mounted. The meeting's record says which resources were available, which were listed, and, when they were retrieved, what the agent answered and what that request cost.

About forty of Biomni's tools call a model themselves, most of them the database tools that turn a question into a query. They need an API key inside the container, which is otherwise given none. Pass the names of the variables to forward, and set Biomni's own settings with `environment`:

```python
executor = session_executor(
    "bio",
    data_lake=lake.directory,
    forward_env=("ANTHROPIC_API_KEY",),
    environment={"BIOMNI_LLM": "claude-sonnet-4-5"},
)
```

A forwarded variable is passed to Docker by name, so its value never appears in a command line, and the record lists only its name. It must be set on your machine, or the session refuses to start. Code in the session can read it, and with the network on it can send it anywhere, so forward only a key you are prepared to have used by the code agents write. Values in `environment` are recorded in full, so they are not for secrets.

`commercial_mode=True` leaves out the data lake files and know-how documents that Biomni's commercial mode leaves out because their licenses forbid commercial use, and describes the data lake as that mode does. `download_data_lake(directory, names=commercial_data_lake())` fetches only the files it uses. It does not check the licenses of the tools themselves or of the databases they query; see `license_info.md` in `virtual_lab/sandbox/biomni_package`.


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

The failure goes to the author rather than to the critic because the author knows what the code was meant to do, while the critic's job is scientific judgement rather than debugging. Each attempt past the first costs another round of code generation, so `max_attempts` defaults to 3. The record written to `executions/<save_name>.json` lists every attempt, the code it ran, and what that produced, alongside the total token usage and cost of the repairs, making the multiplier visible rather than buried.

Repairs take the same limits as a meeting: `max_cost` is checked before each repair request, `max_completion_tokens` caps each one, and `on_usage` and `before_request` let a caller keep a running total or stop the loop. If the loop is stopped partway, by its limit, a request that fails, or Ctrl-C, every attempt it ran and what the repairs cost are recorded under `save_dir/partial/executions/<save_name>.json` before the error propagates.

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


## Running a project within one budget, and resuming it

A project is many meetings, with code run and repaired between them. `Project` holds them to one budget and keeps a ledger, `project.json`, of every step: what it was asked to do, what it cost, how it ended, and where it was saved.

```python
from virtual_lab import CodeArtifacts, DockerExecutor, Project, ProjectBudgetExceededError

project = Project(Path("results/kp3"), goal="Design nanobodies against KP.3.", max_cost=20.0, session=session)

try:
    plan = project.meeting(
        "team", "Choose the computational tools.", name="tools",
        team_lead=principal_investigator, team_members=(immunologist, ml_specialist), num_rounds=2,
    )
    code = project.meeting(
        "individual", "Write the ESM scoring script.", name="esm",
        team_member=ml_specialist, summaries=(plan.summary,), output_schema=CodeArtifacts,
    )
    outcome = project.repair(code.output, ml_specialist, DockerExecutor(), name="esm_run")
except ProjectBudgetExceededError as error:
    print(f"Stopped at ${error.spent:.2f} of ${error.limit:.2f}")
print(project.spent, project.remaining)
```

- `max_cost` covers every meeting and repair in the project, across every run of it on the same directory. Each step is given what is left as its own limit, and the project's total is checked before every request any step sends, so steps run from several threads at once are stopped together. As for a meeting, the limit can be overrun by the cost of one request per step running at the time.
- Each step is recorded as soon as it starts, and what it has cost is updated after every response. A step that fails, or is interrupted by Ctrl-C, is recorded with its error and what it cost. One whose process is killed outright is found running when the project is next opened and is marked interrupted, with what it had cost by its last response.
- A step that finished is never paid for twice. Opening the project again on the same directory and asking for a step under the same name and with the same inputs reads it back from disk, for nothing. A script that stopped partway, for its budget, an error, or Ctrl-C, therefore carries on where it stopped when run again, perhaps with a higher `max_cost`. A failed step is held again.
- Asking for a finished step under its name with different inputs is refused with `ProjectStateError`, which names the inputs that differ, since replacing it could change what later steps were built on; give the new step a name of its own. Agents, tools, and resources are compared by everything the agents are told of them, so a tool described differently or a know-how document edited is a different input. A meeting's transcript, record, and structured output, and a repair's record and code, are checked against hashes in the ledger before they are read back, so a file changed since is refused too. A directory holding a project with another goal is refused.
- Steps are named `meeting_001`, `meeting_002`, and so on, and `repair_001` onwards, in the order they are asked for, unless named. Name every step run from a thread, since that order is not fixed. Meetings and repairs are saved under `meetings/` by name, laid out as `hold_meeting` and `run_with_repair` lay them out.
- Options given to `Project` apply to every meeting, under those given to a meeting. `session`, `chat_models`, and `client` are the project's own, and are used for repairs too. A session does not survive the process, so meetings read back on a later run leave nothing in its interpreter; the files they wrote in its directory remain. Only one process should use a project's directory at a time.

### Letting the principal investigator run the project

`run_project` hands the deciding to the team lead. The team lead chooses a team, unless one is given, the team makes a plan, and then every round the team lead reads what the project has done and decides one next step, restating the plan with each task's status:

- `team_meeting`: a discussion it leads, with the team members and critic it names.
- `individual_meeting`: one team member, or the team lead, works on an agenda, with the critic critiquing.
- `write_code`: one of them writes code, which is run with `executor` and repaired if it fails. Offered only with an executor.
- `change_team`: scientists are brought onto the team, up to `max_team_size`, or let go.
- `finish`: the team lead gives the project's answer. The critic reviews it against the goal and the work done, and the project ends only if the critic agrees; otherwise its objections are what the next round goes on.

```python
from virtual_lab import DockerExecutor, Project, run_project

project = Project(Path("results/kp3"), goal="Design nanobodies against KP.3.", max_cost=20.0)
report = run_project(project, max_rounds=12, executor=DockerExecutor())
print(report.status, report.reason)
print(report.answer)
```

- A run ends `finished` when the critic accepts an answer, `out_of_budget` when the project's `max_cost` runs out, `out_of_rounds` after `max_rounds` rounds, `stalled` after `max_stalled_rounds` rounds in a row without one more of the plan's tasks done, or `stopped` by the approve hook. Running out of budget ends it with a report, not an error.
- Every meeting is told the goal, the team, and the plan as it stands, and is given the summaries of the work so far. A decision that cannot be carried out, such as one naming someone not on the team, is recorded and the team lead is told why in the next round.
- `approve` is called with each round's number and decision before it is carried out, and returns the decision to carry out, which it may change, or `None` to stop the project. With none, the project runs on its own.
- `research_log.json` records every round, with the decision as proposed and as carried out, and every change to the team, and is saved after every round. The run ends with `report.json` and `report.md`: the answer, or the last one the critic did not accept and its objections, the team, the plan, every round, and the cost.
- Every step is a step of the project, so a run that stopped is carried on by running it again on the same directory: the steps it took are read back for nothing. Nothing a step is asked depends on `max_cost`, `max_rounds`, or `max_stalled_rounds`, so they can be raised to carry a project on. For the same reason, the team lead is not told how many rounds or how much money is left. A decision the approve hook approved, changed or not, is not asked about again; one it stopped is.


## Measuring an agent or a team on Biomni's benchmarks

Biomni is measured on three benchmarks, and so can an agent or a team here, on the same questions and by the same rules. Reading them needs `pip install "virtual-lab[eval]"`. Their files, about 4 MB, are fetched once from where Biomni fetches them, under their own licenses, and are not shipped with this package:

```python
from pathlib import Path
from virtual_lab import BiomniEval1Benchmark, HumanitysLastExam, LabBench, download_benchmarks

download_benchmarks(Path("data/benchmarks"))
eval1 = BiomniEval1Benchmark(Path("data/benchmarks"))        # 433 questions in 10 tasks
dbqa = LabBench(Path("data/benchmarks"), "DbQA")             # 60 questions; "SeqQA" has 70
hle = HumanitysLastExam(Path("data/benchmarks"))             # 52 questions
```

| Benchmark | Questions | Scored by |
| --- | --- | --- |
| Biomni-Eval1 | 433, in ten tasks: CRISPR delivery, causal genes from three GWAS sources, variant prioritization, LAB-Bench's DbQA and SeqQA, patient gene detection, rare disease diagnosis, and screen gene retrieval | Each task's own rule, as in Biomni's `BiomniEval1`: a letter, gene, or variant compared as Biomni compares it, a diagnosis by its OMIM ID, and a patient's genes by any overlap. Reported overall and per task. |
| LAB-Bench | DbQA or SeqQA, from the file Biomni's `lab_bench` reads (`subset="test"`), its sampled file, or all of LAB-Bench (`subset="all"`, 520 and 600) | Accuracy, coverage, the share refused, and precision, as Biomni computes them |
| Humanity's Last Exam | The 52 multiple choice questions in biology and medicine that Biomni samples | Accuracy |

Each file is checked against a hash pinned in `virtual_lab.constants` whenever it is read, and one that differs is refused, so two scores made here were made on the same questions. LAB-Bench's options are shuffled exactly as Biomni shuffles them, with a refusal option added, so each question has the letters it has in Biomni and in Biomni-Eval1. Five of the Humanity's Last Exam questions refer to an image that Biomni does not show its agent, and neither does this; their `metadata["has_image"]` is true, so they can be left out.

`run_benchmark` asks each question of an agent or a team, reads the answer out of what was said, and scores it:

```python
from virtual_lab import DockerSession, SingleAgent, TeamMeeting, run_benchmark, session_executor

with DockerSession(Path("results/session"), executor=session_executor("bio")) as session:
    report = run_benchmark(
        eval1,
        SingleAgent(bioinformatician, code_actions="tags"),
        save_dir=Path("results/eval1"),
        extractor="gpt-5.2",
        session=session,
        max_cost=50.0,
    )
print(report.metrics["accuracy"], report.cost)

run_benchmark(hle, TeamMeeting(principal_investigator, (geneticist, scientific_critic), num_rounds=2), ...)
```

- `SingleAgent` holds an individual meeting on each question, with the question as its agenda, and by default no rounds of criticism, which is how Biomni's agent answers. `TeamMeeting` holds a team meeting, one round by default. Both take anything else `hold_meeting` does, such as `code_actions`, `resources`, or `tools`. Any function of a question and a `SolverContext` that returns a transcript, or an `Attempt`, can be the solver instead.
- The answer is read as Biomni reads it: in one more request, a model is told it is "evaluateGPT", given the question as the task's output requirement and the whole transcript as the history, and asked for the answer in the benchmark's schema for the question. `extractor` names that model; Biomni uses the agent's own. It runs at temperature 0, which `extractor_temperature` changes.
- A session is shared by every question, as Biomni's agent keeps one interpreter across its tasks, so a question can find what an earlier one left in it. Pass a function that makes a session, such as `lambda: DockerSession(...)`, for a fresh one per question, closed when the question is done.
- `max_cost` is what the run may spend, and `max_cost_per_question` what one question may. A question is not started once the run's limit is spent, and one running when it is reached is left unfinished. A question that reaches its own limit fails, and the run goes on. Both count the reading of the answer.
- Each question's result is saved under `save_dir/results/` as soon as it is scored, with its answer, score, usage, cost, and where its transcript is. Running again on the same directory carries on where the last run stopped and asks nothing already answered. A directory holding a run of a different benchmark, solver, or extractor is refused, so its results are never mixed with another's.
- A question whose solver fails, or whose answer cannot be read, is saved as a failure and scored 0, and the run goes on. Three failures in a row stop it, since that is more likely a missing key than three hard questions; `max_consecutive_failures` changes that. `retry_failed=True` asks the failed questions again.
- The report, also saved as `summary.json`, holds the benchmark's measures, how many questions were finished and failed, what the finished ones cost, what this run spent, and why it stopped, if it did.

Code written against Biomni's own classes runs against these: `BiomniEval1` has the methods of Biomni's (`evaluate`, `get_instance`, `list_tasks`, `get_task_stats`, `batch_evaluate`, `get_instances_by_task`), and `LabBench` and `HumanitysLastExam` have `get_example`, `get_iterator`, `evaluate`, and `output_class`.

## Querying scientific databases

Tools that look something up go through `virtual_lab.web`, which decides where a request may be sent. An agent chooses the arguments to a tool, which means a model's output determines part of every URL, so the destination cannot be left to the tool:

- Requests are allowed only to the scientific databases named in `ALLOWED_HOSTS`, over HTTPS. Membership is exact, so a host that merely ends in an allowed one (`evilrest.uniprot.org`) and a subdomain of one (`x.rest.uniprot.org`) are both refused, as are a plaintext address, a loopback or link-local address, a non-HTTPS scheme, and a URL carrying credentials.
- The URL is normalised through the HTTP client before it is checked, and the checked URL is the one requested. Validating with one parser and requesting with another is not safe: CPython ends a URL's authority at a slash, question mark, or hash, while urllib3 also ends it at a backslash, so `https://169.254.169.254\@rest.uniprot.org/` reads as an allowed host to the first and as the cloud metadata service to the second.
- Redirects are followed manually and the destination is checked at every hop. A client left to follow redirects itself checks the address it was given, not the address it ends up talking to, which turns one allowed host into a request to anywhere.
- Identifiers are percent-encoded into a fixed template by `build_url`, so an identifier containing `/`, `?`, or `#` becomes part of the path rather than reshaping the request. A segment that is `.` or `..` is refused outright, since a dot needs no encoding and the server would resolve it.
- Responses are capped at `MAX_RESPONSE_BYTES` whether or not the service declares a length, requests to one host are spaced by `MIN_SECONDS_BETWEEN_REQUESTS`, temporary failures are retried with backoff that honours `Retry-After`, and repeated questions are answered from an in-memory cache bounded by both entry count and total size.

```python
from virtual_lab import request_json

data = request_json("https://rest.uniprot.org/uniprotkb/P01308.json")
```

Adding a database means adding its host to `ALLOWED_HOSTS`. This is a deliberate edit rather than a parameter, so that widening what agents can reach is a visible change to the library.

The tests for this layer run offline against a faked transport, since redirects, rate limits, and oversized responses cannot be requested from a real service on demand. The handful that do query the real databases are skipped unless `VIRTUAL_LAB_LIVE_TESTS=1` is set.

## Giving agents tools

`all_tools` returns everything an agent can call. Pass a `save_dir` to include the structure download, which needs somewhere to write.

```python
from virtual_lab import all_tools, run_meeting

run_meeting(
    meeting_type="team",
    agenda="Choose a target epitope for a nanobody against the spike protein.",
    save_dir=Path("results"),
    save_name="epitope_choice",
    team_lead=principal_investigator,
    team_members=(immunologist, computational_biologist),
    # The same two arguments, so downloads land where this meeting's code will run
    tools=all_tools(save_dir=Path("results"), save_name="epitope_choice"),
)
```

| Tool | What an agent uses it for |
| --- | --- |
| `uniprot_search` | Find proteins by name, gene, or organism and get their accessions |
| `uniprot_lookup` | A protein's sequence, function, domains, disulfide bonds, and known structures |
| `pdb_lookup` | An experimental structure's method, resolution, chains, ligands, and chain sequences |
| `alphafold_lookup` | A predicted structure and its pLDDT confidence, where no experimental one exists |
| `fetch_structure_file` | Download coordinates to disk for code to parse |
| `pubchem_lookup` | A small molecule's formula, weight, SMILES, and computed descriptors |
| `chembl_search` | Find drugs and compounds by name and get their ChEMBL identifiers |
| `chembl_lookup` | A drug's clinical status, route, mechanism of action, and drug-likeness |
| `chembl_target_search` | Find a protein target and get the identifier bioactivity is keyed by |
| `chembl_activities` | What binds a target, how tightly, and in what assay |
| `europepmc_search` | Find published articles and biology preprints, with citation counts |
| `europepmc_lookup` | One article's abstract and details, by PMID, PMCID, or DOI |
| `europepmc_fulltext` | Read an open access article, minus its references and front matter |
| `arxiv_search` | Find machine learning and computational preprints, which Europe PMC does not index |
| `pubmed_search` | Abstracts or full text from PubMed Central |
| `data_files` | What data files are in the project directory, and how big |
| `inspect_data_file` | What is in a CSV, TSV, or Excel file, and how it will be read wrongly |

Use `tools_for("uniprot_lookup", "pdb_lookup")` to give a particular agent only part of the set, and `TOOL_REGISTRY` to see what is available. `fetch_structure_file`, `data_files`, and `inspect_data_file` are not in the registry, because none of them can exist until it has been told which directory it may touch; get them from `all_tools(save_dir=...)` or build them with `structure_file_tool(work_dir)` and `data_file_tools(work_dir)`. Each underlying function can also be called directly, which is what tests and analysis scripts do:

```python
from virtual_lab import get_protein

protein = get_protein("P01308")
print(protein.length, protein.pdb_ids[:3])
print(protein.report())
```

Three things about this are deliberate:

**A record and its report are different things.** `get_protein` returns every field worth keeping, while `report()` returns what is worth paying for in a request. A sequence of 35,000 residues is on the record and not in the report, because it would be most of a context window spent on one field.

**Queries are built from named arguments, not written by a model.** A model asked to compose a query payload writes a plausible one whose mistakes come back as an empty result rather than as an error, which is the hardest kind to notice. The cost is a function signature per database, which is also what makes these testable.

**Coordinates go to disk, not into the conversation.** One moderate structure file is several hundred kilobytes, and code an agent writes runs in a sandbox with no network of its own. So `fetch_structure_file` fetches outside the sandbox and writes into `artifacts/<save_name>/structures`, which is inside the one directory mounted into the container, and reports back the path to use from there. A file written anywhere else is invisible to the code that needs it. Where it writes is bound when the tool is built and is not a parameter a model fills in, which is why `all_tools` takes the same `save_dir` and `save_name` that `run_meeting` does.

### What these services will tell you that is not true

Each lookup refuses a particular confident wrong answer, because every one of these arrives as a successful response rather than as an error:

- **PubChem renamed its properties in 2025.** A request for `CanonicalSMILES` is answered with a key called `ConnectivitySMILES`, so code that reads back the name it asked for raises `KeyError` on a perfectly good response. `SMILES` is now the one carrying stereochemistry.
- **PubChem accepts a structure it has never seen**, answering 200 with `CID: 0` rather than an error. Only the identifier says nothing was found.
- **A SMILES cannot go in the URL path.** It carries `/` and `\` for stereochemistry, and percent-encoded into a path segment PubChem answers 400. Since PubChem keeps 400 for a request it could not parse, treating it as "no such compound" would report a real compound as nonexistent whenever a URL was built wrongly, so only 404 means missing here.
- **ChEMBL sends its numbers as strings**: `max_phase` arrives as `"4.0"` and molecular weight as `"180.16"`.
- **`max_phase` is -1 for 2,499 molecules** whose clinical status ChEMBL does not know, which sorts below a compound that was never tested.
- **Some ChEMBL measurements lost their units in standardisation.** The most potent IC50 against EGFR by raw value reads `5.012E-9 nM`, a hundred million times tighter than any real binding; its original value was 17.3 in units the record no longer names. Those rows are exactly the ones with no pChEMBL value, so `chembl_activities` requires one and orders by it.
- **Searching a gene symbol ranks the wrong thing first.** `EGFR` returns two protein-protein interactions and mouse EGFR above human EGFR, and restricting the organism alone does not fix it, which is why the target search tool passes `single_proteins_only` on by default. The library function it calls leaves the filter off unless asked, so a caller looking for a complex can still find one.

And the two literature searches read the same query in opposite ways, which is the one to know about:

- **arXiv reads loose words as alternatives.** `protein language model` sent as written returns 1,285,358 results, being everything mentioning any of the three words. Requiring all three returns 893 and the quoted phrase returns 344. The count misleads as much as the results do, so `arxiv_search` joins loose terms with `AND`, says in the report that it did, and passes through any query that already uses a field prefix or an operator.
- **Europe PMC reads the same words as requirements**, so a long query there finds less rather than more. Opposite defaults, same phrasing from an agent.
- **Europe PMC does not index arXiv**, which is why there are two searches rather than one. Most computational method work is only on arXiv, and much of it stays there.
- **A quoted identifier matches nothing.** `PMCID:"PMC3258128"` and `EXT_ID:"21937511"` both return zero hits while the same queries unquoted return the article, but a DOI needs the quotes because it carries slashes. The two that go unquoted are checked against their shape first, which is what makes that safe.
- **An unknown article is a 500, not a 404.** Asking Europe PMC for the full text of an article it does not hold gets the same status as an outage, so the retry logic spends three attempts and several seconds and then blames the service. `europepmc_fulltext` reads the record first, which says plainly whether the text exists.
- **`isOpenAccess` is the letter `Y` or the letter `N`**, as are ten other fields on the same record. `"N"` is a non-empty string, so every one of them is true when tested directly.
- **References are about 28% of an article's body text** and are a list of other people's titles, which reads to a model as though this article had discussed all of them. They are left out, along with funding and author contributions, and the report says which sections were dropped.
- **ElementTree expands internal entities.** That was checked rather than assumed. Neither service has any reason to send a `DOCTYPE`, so one is refused before parsing rather than the expansion being bounded.

**PubChem needs a particular TLS handshake.** Its edge answers this library's ordinary handshake with 503 and a "server too busy" body, while reporting the service as healthy to `curl` from the same address in the same second. `web.HOST_CIPHERS` offers it a cipher list it accepts. The connection is still TLS 1.3 with certificate verification on; only the list offered in the handshake differs. No other service needs this.

### Reading the project's own data

The tools above fetch what is known. A project also has data of its own: a spreadsheet from a collaborator, a table downloaded by hand, a file written by the last piece of code that ran. An agent that cannot see inside those files writes code against the columns it imagines are in them.

`inspect_data_file` answers the first question about a file, which is always the same question, without a round trip through the sandbox. It reads `.csv`, `.tsv`, `.tab`, `.txt`, `.xlsx`, and `.xlsm`, reports the shape and what each column holds, and shows the first few rows. It does not return data for computing with; the sandbox is still where analysis happens.

Excel files are read without a dependency, straight out of the archive, so there is nothing to install and nothing that runs while a file is being opened.

The filename comes from a model, so it is treated the way a filename a meeting asks to write is: checked for traversal, resolved, and checked again against the directory it must be inside. Resolving first also settles symbolic links, so a link pointing out of the directory is refused rather than followed.

### What a data file will tell you that is not true

The point of reading the file before the code is written is not the shape. It is that a table misleads quietly, and every one of these is a file that loads without complaint:

- **One `<0.001` makes the whole column text.** A measurement below the assay's detection limit is written as a bound, and a single one of them turns a column of numbers into a column of strings. Every mean taken afterwards is wrong or skipped, and the values dropped are not missing at random: they are the strongest and weakest samples.
- **A spreadsheet turns gene symbols into dates.** SEPT2 becomes 2-Sep as it is typed, and the symbol cannot be recovered from the file afterwards. A column holding both dates and text is the signature, so that is what is reported.
- **A date in a spreadsheet is an ordinary number.** `45902` is 2025-09-02 only if you follow the cell's style to its number format, and Excel records no flag saying a format means a date. Without that chase a corrupted gene column reads as a column of five digit integers.
- **A byte-order mark renames the first column.** Read as `utf-8` rather than `utf-8-sig`, the header is `\ufeffgene` and never matches `gene`.
- **Two columns with the same name leave one of them gone.** Readers that key rows by name keep the last, without a word.
- **Identifiers lose their leading zeros.** `00123` read as a number is `123`, and stops matching anything.
- **A row in a spreadsheet holds only its non-empty cells.** Taken in order rather than by cell reference, every value after a blank moves one column left, which produces a table that is wrong and looks right.
- **The cell reference is optional.** Writers that stream cells out in order leave it off, and a reader that places cells only by reference reads the whole sheet as empty. A cell without one goes in the column after the one before it, as Excel reads it.
- **A cell's text can be split into runs.** Asking for the first `<t>` under it returns nothing at all, because the runs are between them.
- **`float()` accepts more than anyone means by a number.** `float("1_000")` is 1000.0, `float("nan")` and `float("inf")` both succeed, and `float("١٢٣")` is 123.0. A column whose missing values are spelled `nan` is otherwise reported as numeric.
- **`csv.Sniffer` cannot read a one-column file** and raises rather than saying so, which a list of gene names is.
- **cp1252 decodes every possible byte**, so it never fails and a wrong guess at the encoding is silent. UTF-8 first is the only ordering that is ever right.
- **A 204 KB spreadsheet can declare a 200 MB sheet.** That was measured. The size is recorded inside the archive by whoever wrote it, so the cap is on the bytes actually unpacked.
- **A cell's position costs nothing to write.** A 5.7 KB sheet with one cell in column XFD of each row pads out to 600 MB, which no byte cap sees. Rows are held only as far as their last value, the cap is on the cells held, and a column past XFD is refused as damage.

A note on `uniprot_search`: `reviewed_only` is off by default. Curation covers a small fraction of UniProt, and for some classes of sequence it covers none of it, so a search for a nanobody with it on returns proteins whose reference titles mention one while filtering out every real camelid VHH.
