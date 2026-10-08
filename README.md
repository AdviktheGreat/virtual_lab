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

Tools from MCP servers, and serving tools over MCP with `virtual-lab-mcp`, need the MCP SDK, which the `mcp` extra installs: `pip install "virtual-lab[mcp]"`. Saving meetings as PDFs needs the `pdf` extra and the Pango library (see [Saving a meeting or a project as a document](#saving-a-meeting-or-a-project-as-a-document)). The web interface needs Gradio, which the `ui` extra installs: `pip install "virtual-lab[ui]"` (see [The web interface](#the-web-interface)). Reading papers in PDFs needs pypdf, which the `papers` extra installs: `pip install "virtual-lab[papers]"` (see [Growing the toolset from papers](#growing-the-toolset-from-papers)).


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

Keys can be kept in a `.env` file, one `NAME=value` per line. As Biomni does, `import virtual_lab` reads the `.env` file in the directory Python was started in, if there is one, and sets each variable that is not already set, so a key set in the shell outranks the file's. It prints nothing. `load_env("keys.env")` reads another file, `load_env(override=True)` replaces what is set, and each returns the names of the variables it set. Setting `VIRTUAL_LAB_LOAD_ENV=0` stops the file being read on import. What is read reaches this process alone: a session's code gets a variable only when it is forwarded by name (see `forward_env` below).


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

### Following a meeting as it happens

Biomni's agent yields each step of its work once the step is done, which is what its web interface shows. A meeting does the same through `on_event`, which is called with a `MeetingEvent` for everything that happens, and with `stream=True` it is also told of each reply as it is written, word by word:

```python
from virtual_lab import MeetingEvent, hold_meeting

def show(event: MeetingEvent) -> None:
    if event.kind == "turn":
        print(f"\n--- {event.speaker}, round {event.round} ---")
    elif event.kind == "writing":
        print(event.text, end="\r")
    elif event.kind == "cell":
        print(f"Ran {event.data['language']} ({event.data['status']}), figures: {event.data['plot_paths']}")
    elif event.kind == "usage" and event.data["cost"] is not None:
        print(f"Spent so far: ${event.data['cost']:.4f}")

hold_meeting(..., on_event=show, stream=True)
```

| Kind | When | What it holds |
| --- | --- | --- |
| `started` | The meeting begins | `data`: the meeting type, agenda, questions, rounds, team, tools, `max_cost`, and session |
| `resources` | The resources of Biomni's environment are chosen | `data`: what the record says of them |
| `turn` | An agent's turn begins | `speaker`, `round`, and `data`: its name and model |
| `writing` | More of a reply is written, with `stream` | `text`: the reply so far, which replaces the last with the same `data["request"]` |
| `message` | A message is added to the transcript | `speaker`, `text`, and `data`: its kind (`prompt`, `response`, `code_action`, `code_output`, `tool_output`, or `structured_output`) and its index |
| `tool_calls` | Tools are about to run | `data["calls"]`: each call's id, name, and arguments, in full |
| `code` | Code from `<execute>` tags is about to run | `data`: the code and its language |
| `cell` | Code ran in the session, from tags or `run_code` | `data`: the run, as `CellResult.to_dict` gives it, and `plot_paths`, where its figures are |
| `usage` | A response came back | `data`: the usage so far, with its `cost`, or `None` if unknown |
| `finished` | The meeting is saved | `text`: the summary, and `data`: where it was saved, the structured output, the usage, and how long it took |
| `failed` | The meeting stopped with an error | `text`: the error, and `data`: its type, where the partial transcript is, and the usage |

- An exception `on_event` raises stops the meeting the way any other failure does, saving what was done under `save_dir/partial/`, even while a reply is being written. This is how a person can stop a meeting part way through a reply. One it raises when told the meeting finished, which is saved by then, or failed, which would hide the error that ended it, is only warned about.
- With `stream`, every reply is asked for with its usage, so what it cost is still known. OpenAI-compatible servers that leave the usage of a streamed reply out make the cost unknown, as they would without streaming, and one that refuses to be asked fails the request; a model told not to stream its usage (`stream_usage=False`) is left as told. A model that cannot stream sends its reply whole, as one `writing` event. Structured outputs, and the request that chooses resources, are not streamed.
- `Project` takes `on_event` and `stream` as options for its meetings, and neither makes a finished meeting a different one, so a project carried on with or without them reads its meetings back. A meeting read back is told of by one `read_back` event, with its summary and where it is saved.

### Steering a meeting as it happens

A person following a meeting can steer it as well as watch it. `steer` is called before every agent's turn with a `NextTurn`, which names the meeting, the round, and the agent about to speak, and returns a note or `None`:

```python
from virtual_lab import NextTurn, hold_meeting

def steer(turn: NextTurn) -> str | None:
    if turn.speaker == "Scientific Critic":
        return "Check the controls in the binding assay before anything else."
    return None

hold_meeting(..., steer=steer)
```

- A note is added to the discussion after the agent's prompt, introduced as from the human researcher overseeing the meeting, so the agent about to speak reads it last, and everyone after it reads it too. It is kept in the transcript as from `Human researcher`, with the kind `note` in the record and in the `message` event, and saved documents show it, so no agent may have that title. A note that is empty or only spaces is not added.
- The meeting waits while `steer` does, which is how it is paused: a function that waits for a person to resume holds the meeting before the turn, with nothing sent. `steer` is asked only between turns, so an agent running code or calling tools finishes its turn first.
- An exception `steer` raises stops the meeting the way any other failure does, saving what was done under `save_dir/partial/`.
- `Project` takes `steer` as an option for its meetings, which with `run_project` includes the meetings in which the team lead decides each step, so a note can reach the team lead before its next decision. Like `on_event`, it does not make a finished meeting a different one: what was said is in the meeting's transcript, and a project carried on reads it back.

### Saving a meeting or a project as a document

Biomni saves its agent's conversation as a PDF. A meeting can be saved as one too, or as a page of HTML, laid out as a meeting: its agenda, team, and summary first, then the discussion round by round, each agent in a colour of its own, with the code each agent ran, what it printed, and the figures it drew, then what the meeting cost. A project is saved as its report, with every meeting it finished after it if asked for, each from a new page.

```python
from virtual_lab import save_meeting_html, save_meeting_pdf, save_project_pdf

save_meeting_pdf("results/discussion.json")            # results/discussion.pdf
save_meeting_html("results/discussion.json")           # results/discussion.html, which needs no extra
save_project_pdf("project", meetings=True)             # project/report.pdf
```

- A PDF needs WeasyPrint, which `pip install "virtual-lab[pdf]"` installs, and the Pango library it draws text with, which pip cannot: `brew install pango` on macOS, or `apt install libpango-1.0-0 libpangoft2-1.0-0` on Debian and Ubuntu. Without them a PDF is refused with `PDFExportError`, saying what to install, and HTML can still be saved.
- Everything is read from what was saved, so any meeting can be saved as a document at any time, including one that failed, from `partial/`, and a transcript saved before meetings kept records, which is shown from the transcript alone.
- The prompts each agent was given are left out, since the agenda stands for them; `include_prompts=True` shows them. Notes a person gave through `steer` are always shown. A tool's or the session's output is cut to `max_output_chars`, 6,000 characters by default, from the middle.
- What the agents wrote is rendered as Markdown, with any HTML in it shown as text, and the document carries its figures, so neither the HTML nor the PDF fetches anything: an image an agent linked to, on the web or on disk, is shown as a link to it. A figure is read only from the session's own directory.
- `meeting_html` and `project_html` return the document rather than saving it.


## The web interface

Biomni has a Gradio page to work an agent from. The Virtual Lab has one for a lab: set up a meeting or a project, follow it as it happens, steer it or let it run, and read back everything it has done. It needs Gradio, which the `ui` extra installs.

```bash
pip install "virtual-lab[ui]"
virtual-lab-ui                          # http://127.0.0.1:7860, opened in your browser
virtual-lab-ui --workspace ~/lab --port 8000 --no-browser
```

```python
from virtual_lab.ui import launch_ui

launch_ui(workspace="~/lab", password="a secret")   # waits until stopped; block=False to carry on
```

The page has four tabs, in a light "lab notebook" look or a dark "control room" one, which the header switches between. It starts as your system is.

- **Meeting room**: the agenda, the questions, who leads, who is on the team, the rounds, the model, the budget, the databases and MCP servers the agents may use, where their code runs, and what they are told of Biomni's tools, data, and know-how (`resources`, as `hold_meeting` takes it, which matters where code runs). The discussion appears reply by reply, as `on_event` reports it, and word by word if asked. While it runs, a note can be added for the next agent to speak (as `steer` takes it), the meeting can be paused, and it can be stopped, which saves what was done under `partial/`. A finished meeting can be followed up by a new one that builds on its summary, and saved as HTML or PDF. Scientists of your own can be added to the lab.
- **Project**: a goal, and the team lead chooses the team (or you do), makes a plan, and decides each step. By default each decision waits on the page for you: approve it, approve it with your changes to the step, or stop the project, at once or after the step under way so that it ends with a report. Letting the project run on its own approves every step, still within its budget. The report can be saved as HTML or PDF, with its meetings if asked for.
- **History**: every meeting and project in the workspace, read from the files they were saved in, including those run from a script and those that were interrupted. Selecting one shows it as a document. A project that stopped can be carried on from here, with a larger budget and more rounds.
- **Settings**: the model, whether to stream, the budget, where code runs (nowhere, in a Docker container of plain Python or one of Biomni's three environments, or on this machine with no isolation), the interpreter for code on this machine, and a file of MCP servers to offer. These are what each new form starts with.

Anything that waits for a person waits on the page: a step to approve, a tool that needs approval (see [Tools that wait for a person's approval](#tools-that-wait-for-a-persons-approval)), and a question an MCP server asks.

- A run goes on whether or not a page is open, and a page opened later, or reloaded, shows the run that is going on. One meeting and one project can run at once.
- Everything is kept in the workspace, `~/virtual_lab_workspace` unless `--workspace` says otherwise: `settings.json`, `scientists.json` for the scientists you add, and a directory for each meeting in `meetings/` and each project in `projects/`, named for the date and the agenda or goal, and saved as `hold_meeting` and `Project` save them. The page shows only what is in those files, so a run started from a script into the workspace appears in the history.
- The page listens on this machine alone, with no password. Anyone who reaches it can spend on your API keys and, if code runs, run code on your machine or in its containers, so `--share` (a public link through Gradio, for 72 hours) and any `--host` beyond this machine are refused without a password, from `--password` or `$VIRTUAL_LAB_UI_PASSWORD`. Sign in with that password and any name.
- The keys are the ones the models need, as [Models and API keys](#models-and-api-keys) says. A key can be set in Settings too, and is then used until the interface stops, never saved or shown. A model whose key is not set is refused when the run is started, saying which.
- What the agents wrote is rendered as Markdown with any HTML shown as text, and a link opens in a new tab only if it is to a web page.


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
- Every meeting is told the goal, the team, and the plan as it stands, and is given what the work so far found, as `memory` says (below). A decision that cannot be carried out, such as one naming someone not on the team, is recorded and the team lead is told why in the next round.
- `approve` is called with each round's number and decision before it is carried out, and returns the decision to carry out, which it may change, or `None` to stop the project. With none, the project runs on its own. A `steer` given to the `Project` adds a person's notes to its meetings as they go (see [Steering a meeting as it happens](#steering-a-meeting-as-it-happens)).
- `on_event` is called with a `ProjectEvent` when the team is chosen or changed (`team`), the plan is made (`plan`), each step is decided (`decided`, before `approve` is asked), code is run (`code`), each round ends (`round`), and the project ends (`finished`, with the report), and with every `MeetingEvent` of every meeting it holds, so a project can be followed as it happens. The `Project`'s own `on_event`, if it has one, is called first. An exception it raises stops the project like any other error, except when it is told the project ended, once the report is saved, which is only warned about.
- `research_log.json` records every round, with the decision as proposed and as carried out, and every change to the team, and is saved after every round. The run ends with `report.json` and `report.md`: the answer, or the last one the critic did not accept and its objections, the team, the plan, every round, and the cost. A run that fails with any other error leaves no report, and the log says what the error was.
- Every step is a step of the project, so a run that stopped is carried on by running it again on the same directory: the steps it took are read back for nothing. Nothing a step is asked depends on `max_cost`, `max_rounds`, or `max_stalled_rounds`, so they can be raised to carry a project on. For the same reason, the team lead is not told how many rounds or how much money is left. Every decision carried out, whether the approve hook changed it or there was no hook, is kept in the log and not asked about again, however far a later run gets before it stops; one the hook stopped is asked about again.

#### What each step is given of the work before it

Passing every meeting the summary of every meeting before it, as the Virtual Lab does, fills the context as a project grows, with much that the meeting at hand does not need. So `run_project` keeps what the work established in a `LabMemory`, as findings: each meeting restates what it found as `Findings`, one claim and its evidence apiece, in one more request, and after code is run its author says what the run established, in a short meeting of its own. Each finding gets an id, `F1`, `F2`, and so on. What a step is given then depends on `memory`:

- `"pick"`, the default: the team lead is shown every finding by its id and claim, and what each step found, with the latest step in full, and names in each decision the findings the step needs. The step is given those in full and no others; for `finish`, the critic is given the ones the answer rests on. This is how Biomni's agent picks the tools, data, and know-how a task needs, and costs no requests beyond the findings themselves. A decision naming a finding that does not exist is refused.
- `"bm25"`: each step is given the findings whose claim and evidence best match its agenda and questions, or a proposed answer, by BM25, at most `findings_per_step` of them.
- `"summaries"`: no findings are kept, and each step is given the summary of every step before it, as before.

The findings are saved in `memory.json` and listed in the report, and the research log records which findings each round was given and which it made. A memory can be used on its own too:

```python
from virtual_lab import Finding, LabMemory

memory = LabMemory()
memory.add([Finding(claim="Nb21 binds KP.3 at 2 nM.", evidence="SPR, three replicates.")], source="assay")
memory.search("KP.3 binding", limit=5)  # by BM25
memory.get(["F1"])
memory.save(Path("memory.json"))
```


## Growing the toolset from papers

Biomni grows its toolset by reading papers for the computational tasks, databases, and software in them. `read_paper` and `extract_paper_findings` do the same reading. A PDF needs pypdf: `pip install "virtual-lab[papers]"`. Plain text, Markdown, and LaTeX files need nothing.

```python
from virtual_lab import extract_paper_findings, read_paper

reading = extract_paper_findings(read_paper("paper.pdf"), model="gpt-5.2", max_cost=1.0)
for task in reading.findings.tasks:
    print(task.task_name, "|", task.inputs, "->", task.outputs)
print(reading.chunks, reading.cost, reading.failed_chunks)
```

- The text is cut into chunks of 4,000 characters with 400 repeated between them, as Biomni cuts it, at paragraph breaks where it can, then line breaks, sentences, and words. Each chunk is read with Biomni's guidelines: only tasks that are common across biomedical research, with clear inputs and outputs, implementable with Python or Linux code, and named with their exact methods. Then the findings of every chunk are merged, one to each name, and one more request applies the same filter to the paper as a whole.
- Every answer is asked for in a schema (`PaperFindings`: tasks, databases, and software, each with the fields Biomni records), and one that does not fit it is not used. Biomni reads each chunk as free text and parses the consolidation's reply as JSON, and when it cannot, returns the raw reply as a single "task" with an `error` field. Here a chunk the model refuses or cannot answer in the schema is recorded in `reading.failed_chunks` with why, and the rest are used; a paper none of whose chunks can be read is refused with `PaperReadingError`.
- Findings too many for one request, more than `max_consolidation_chars`, are consolidated in batches and then again as a whole, where Biomni joins every chunk's reply into one request, which fails once they outgrow the model's context. If the consolidation fails, `PaperReadingError` holds what the chunks found, merged but not filtered, in its `findings`.
- A paper is read as far as `max_chars` (200,000, as Biomni reads), cut at the end of a sentence, and `reading.truncated_from` says how long it was. A file over 100 MB, a PDF over 500 pages, a PDF that needs a password to open, and a file that is not a PDF, text, or Markdown are refused. A PDF locked only against copying and printing, as many publishers' are, is read.
- `max_cost` is checked before every request and stops the reading with `PaperBudgetExceededError`, which holds in `findings` what the chunks read so far showed, unfiltered, so that what was paid for is not lost. To keep one limit across many papers, pass the same `MeetingUsage` as `usage` to each. A model whose price is not known cannot be given a limit.
- A PDF's text is as pypdf finds it, so the columns of a two-column paper and the labels of its figures can be mixed into the prose, as in what Biomni reads. Read the tasks as suggestions to check, not as facts about the paper.

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

### Your own tools

`tool_from_function` makes a tool of any Python function, the way Biomni's `add_tool` does. Biomni asks a model to read the function's source and write out its parameters. Here the parameters come from the signature and its type hints, and the descriptions from the docstring, so nothing is invented:

```python
from virtual_lab import tool_from_function

def gc_content(sequence: str, window: int = 100) -> list[float]:
    """The GC fraction of a DNA sequence, in windows along it.

    :param sequence: The sequence, as A, C, G, and T.
    :param window: How many bases each window spans.
    """
    ...

tool = tool_from_function(gc_content)
hold_meeting(..., tools=(tool,))     # the agents call it as a tool
```

A parameter without a default is required, and each is typed by its hint: `str`, `int`, `float`, `bool`, `list[str]`, `dict[str, float]`, `Literal["a", "b"]`, an `Enum`, a `Path`, a `date`, a pydantic model, or anything else pydantic can read from JSON. The arguments a model sends are converted to those types before the function is called, so `"5"` becomes `5` and `{"start": 1, ...}` becomes the model it describes. Arguments that are wrong are not passed on: the model is told what was wrong and can correct the call. The docstring can be in the reST, Google, or NumPy style. `name=` and `description=` override what the function says of itself. A `functools.partial` leaves the arguments it binds out of the tool, and an async function is waited for. A parameter whose type cannot come from JSON, such as a pandas DataFrame, is refused, since no model can pass one; take a path to the file instead.

A function with no docstring needs `description=`, or a model to describe it: `tool_from_function(f, model="gpt-4o")` asks it once, from the function's source, and prints what it wrote. Only the descriptions come from the model; the parameters still come from the signature. A model can describe the same function differently each time, and a project resumed with a tool described differently holds its meetings again, so keep a description you are happy with by adding it to the function as its docstring.

Give the tool to a session instead, and code in the session calls it like any other function, without importing it, as Biomni's agent calls the tools added to it:

```python
with DockerSession(Path("results/session"), tools=(tool,)) as session:
    session.run("profile = gc_content(sequence, window=50)")
    hold_meeting(..., session=session)  # the agents are told of gc_content, and their code calls it
```

The function runs on your machine, outside the container. When code in the sandbox calls it, the call comes out over the session's own connection as JSON, the function runs here, and what it returns goes back the same way. A tool can therefore use what the sandbox does not have: the network when the session has none, an API key the sandbox is never given, a GPU, a licensed program, or files outside the session's directory. That is also the risk: a tool can do whatever the function does, with your permissions, on arguments that code written by a model chose. Give a session only tools you would let that code call. Arguments and results are JSON. NumPy arrays, sets, and paths are converted, and anything else is refused with advice to convert it or pass a file instead. A failure in the tool raises `HostToolError` in the code. A call counts towards the time limit of the code that made it: code stopped at its limit while a tool runs keeps its session, and the tool is left to finish, its result unused. Each call is recorded with the cell that made it: the tool, its arguments, how it ended, and how long it took. The tools are listed to a meeting's agents ahead of Biomni's resources, as functions added for this work that they should prefer, again as Biomni lists the tools added to its agent. Changing a session's tools changes the inputs of every project step held with it. R and bash code cannot call them, and a tool cannot run code in the session whose code called it.

### Your own data and software

A session can also be given data and software of your own, in the form Biomni's `add_data` and `add_software` take them: a dict of each path, or each name, with what it is.

```python
with DockerSession(
    Path("results/session"),
    data={"inputs/counts.csv": "Read counts per gene and sample, from the 2024 RNA-seq run.",
          "inputs/images": "Microscopy of the knockout lines, one TIFF per field."},
    software={"pydeseq2": "Differential expression, a Python port of DESeq2."},
) as session:
    hold_meeting(..., session=session)
```

Biomni lists a file it was given by its name alone, and leaves the agent to find it. Here each file or directory is mounted read-only in the container at `/data/` and its name, `/data/counts.csv` and `/data/images` above, and the agents are told that path. A `LocalSession` reads the data where it is, and tells the agents that path instead. The data must exist when the session is made, no two pieces may share a file name, and in a `DockerSession` none may be in the session's directory, where code could change it, or hold it.

Software is not installed for you: it must be in the image, or in the `python` a `LocalSession` runs. A meeting first looks for each piece in the session, as a Python distribution, a module Python can find, an R package, or a command. Anything it cannot find is warned about and noted in the record as `software_not_found`, and the agents are still told of it, but as not found, since a name can differ from anything the check looks for.

The data and software are listed after the session's own tools, ahead of Biomni's resources, as added for this work and to be preferred, as Biomni lists what is added to its agent. Changing them changes the inputs of every project step held with the session. Data counts by name and description, not by where it is on your machine or what the files hold, so a project resumed with different contents under the same name does not notice the change.

### Tools from MCP servers

`connect_mcp` makes tools of the tools of [MCP](https://modelcontextprotocol.io) servers, as Biomni's `add_mcp` does, from the same YAML config, or from the JSON config that Claude Desktop, Claude Code, Cursor, and others write. It needs the MCP SDK: `pip install "virtual-lab[mcp]"`.

```yaml
mcp_servers:                      # or mcpServers, as Claude Desktop's config has it
  genes:                          # started here, and spoken to over its stdin and stdout
    command: ["python", "-m", "gene_server"]
    env: {GENE_API_KEY: "${GENE_API_KEY}"}
  search:                         # reached at a URL, over streamable HTTP
    url: "https://search.example.org/mcp"
    headers: {Authorization: "Bearer ${SEARCH_TOKEN}"}
```

```python
from virtual_lab import connect_mcp

with connect_mcp("mcp_config.yaml") as mcp:
    hold_meeting(..., tools=mcp.tools)                   # the agents call them as tools
    with DockerSession(Path("results/session"), tools=mcp.tools) as session:
        hold_meeting(..., session=session)               # or code in the session calls them
```

Each tool is named after its server and itself, `genes_lookup` for the tool `lookup` of the server `genes`, so that two servers' tools never share a name, with anything other than letters, digits, and `_` made `_`, so that code can call it. Its description and parameters are the ones the server gives it, and the server checks the arguments. `mcp.servers["genes"]` holds one server's tools.

A command is a list of the program and its arguments, as Biomni's config has it, or a program with its `args`, as Claude's has it. A server started here gets `env` as its environment, with a few variables of this process, such as `PATH` and `HOME`, and not the rest, so an API key reaches only the servers given it; `cwd` is where it starts. A server at a URL gets `headers` with every request, and one whose `type` is `sse`, or whose URL ends in `/sse`, is spoken to over the older SSE transport. `${NAME}` anywhere in these is replaced with the environment variable `NAME`, and `${NAME:-default}` with the default where it is not set. Errors, and what a server wrote to stderr, show `${NAME}` where its value would be, so a key in a URL or a header is not repeated in a traceback; what a tool returns is left as it is. A server with `enabled: false` or `disabled: true` is left out, and `connect_mcp(config, servers=["genes"])` connects to only the servers named. `tools: [lookup, search]` offers only those of a server's tools; an entry in Biomni's form, with `biomni_name` and `description`, gives that tool its description, and the parameters it lists are not used, since the server declares them.

Biomni starts a server again for every call to one of its tools, over stdio only, and returns the first piece of whatever the tool returned, an error included. Here:

- **A server is started once**, and kept running until the tools are closed, so what it loads stays loaded, and calls to it run at the same time. Any left open are closed when Python exits.
- **What a tool returns is given back whole.** Where the server gives a structured result, that is what code receives, such as a dict, and otherwise its text. An image, a recording, or a file in it is described, since none can be shown as text. A model is shown either as text.
- **A failure is a failure.** A tool the server says failed raises `MCPToolError` with what the server said, which the agent is told as an error, and can correct its call from. A call that takes longer than `timeout`, 300 seconds unless given, raises `TimeoutError`, and the server is kept.
- **A server that stops is started again**, or connected to again, at the next call to one of its tools, with a warning, since whatever it held is gone. Only a call that was running when it stopped fails, raising `MCPServerError`.
- **A config that cannot work fails at once.** A variable that is not set is an error, rather than an empty value, and so is a server that does not start, or a tool asked for that a server does not have. What a server writes to stderr is kept out of the way, and its end is shown when the server fails. Where one server fails, those already started are stopped again.

A server's tools run wherever the server does: one started here runs on your machine, with your permissions, on arguments a model chose, even when the code calling it is in a sandbox, so give a session only the servers you would let that code use. Changing a server's tools, or what it says of them, changes the inputs of every project step held with them.

#### Tools that wait for a person's approval

A tool that spends money, or does what cannot be undone, can be made to wait for a person to approve each call to it. A server's `approval` lists those tools by their names on the server, or by patterns such as `create_*`, or lists under `ask` the tools that wait and under `allow` those of them that do not after all:

```yaml
mcp_servers:
  lab:
    url: "https://lab.example.org/mcp"
    approval: {ask: ["*"], allow: ["list_*", "get_*"]}   # every tool waits but those that only read
    instructions: "Estimate an experiment's cost before creating it."
```

`{ask: ["*"], ...}` also holds for any tool the server adds later, which a list of names would let through. Before each call to a tool that waits, the person at the terminal is shown the tool and its arguments and asked to approve it. Where there is no terminal, as in a script run in the background or one whose stderr goes to a file, every such call is declined. A call made by code in a session that has stopped, such as by running out of time, before the person answers is not made, even if they approve it, since no one would receive its result. `connect_mcp(..., approve=decide)` decides another way: `decide` is called with an `ApprovalRequest`, and the call is made only if it returns `True`. A call that is not approved is not made. It raises `ApprovalDeclined`, which the agent is told of, along with not to make it again unchanged. A name in `ask` that the server has no tool of is warned of, since a call to the tool meant would not wait. The descriptions of the tools that wait say so, so that agents know.

A server can also ask a person something while one of its tools runs, as Proto's server asks before it deploys a tool. Its question is asked at the terminal, a form field by field, or, where it asks the person to go to a web page, by opening the page. `answer=` answers another way: it is called with a `ServerQuestion` and returns the answers by field name, `{}` once the person has gone to the page, or `None` to decline. Where there is no terminal, every question is declined. A question whose tool call has stopped waiting, such as by running out of time, is withdrawn, and the answer is not used.

What a server's `instructions` say, and what the server says itself of how to use its tools, are told to the agents of a meeting given its tools, or a session given them.

#### Paperclip, Adaptyv, and Proto

Three services built for agents doing biology have presets, a server's entry written out already, with what waits for approval and what the agents are told:

| Preset | Service | Needs |
| --- | --- | --- |
| `paperclip` | [Paperclip](https://paperclip.gxl.ai): searching and reading the full text of papers, trials, patents, and regulatory documents | An API key in `PAPERCLIP_API_KEY`, from https://paperclip.gxl.ai/keys, or signing in with `auth: oauth` |
| `adaptyv` | [Adaptyv's Foundry](https://foundry.adaptyvbio.com): proteins made and tested in a lab, which costs money | An API token in `FOUNDRY_API_TOKEN`, from the Foundry portal, or signing in with `auth: oauth` |
| `adaptyv_testing` | Adaptyv's testing sandbox, where nothing is made or charged for | A token in `FOUNDRY_TESTING_TOKEN`, from https://foundry.testing.adaptyvbio.com, or signing in with `auth: oauth` |
| `proto` | [Proto](https://github.com/evo-design/proto-tools): over a hundred tools for protein and sequence design, run on your own Modal account | `uv tool install "proto-tools[mcp] @ git+https://github.com/evo-design/proto-tools.git"`, then `modal token new` |
| `proto_hosted` | Proto's hosted server, at `https://mcp.evodesign.org/mcp`: the same tools, with nothing installed here | Signing in with a Proto account, from https://proto.evodesign.org |

```python
with connect_mcp(presets=["paperclip", "adaptyv_testing"]) as mcp:
    hold_meeting(..., tools=mcp.tools)

with connect_mcp(presets={"paperclip": {"auth": "oauth"}, "proto_hosted": {}}) as mcp:
    hold_meeting(..., tools=mcp.tools)                   # signed in to, with no API keys
```

Or a config names one with `preset`, and anything else its entry gives is used in place of the preset's, such as `lab: {preset: adaptyv, approval: ["*"]}` for every one of Adaptyv's tools to wait. The keys and tokens can be in the environment or in the `.env` file, and where one is missing, the error says where to get it. Every one of Adaptyv's tools waits for approval but those that list, get, or estimate a cost, so no experiment is created, submitted, or paid for without a person approving it. Proto's tools that search, describe, and run deployed tools do not wait, but `deploy_tool`, which builds a tool on Modal at your expense, does, as does any other tool either server has or adds. `MCP_PRESETS` lists the presets, with what each needs.

#### Signing in to a server

A server at a URL whose entry has `auth: oauth` is signed in to in your browser, the first time it is connected to, rather than sent an API key. Proto's hosted server is always signed in to; Paperclip's and Adaptyv's can be, in place of their keys, which a preset signed in to is then neither sent nor needs set:

```yaml
mcp_servers:
  proto: {preset: proto_hosted}
  papers: {preset: paperclip, auth: oauth}
  search:
    url: "https://search.example.org/mcp"
    auth: oauth
```

`connect_mcp` finds the server's authorization server, registers with it, and opens its sign-in page, with the MCP SDK's OAuth client, which uses PKCE. Once you have signed in, the browser is sent back to `http://127.0.0.1` on a port of this machine's, where the sign-in is received. The tokens are kept in `~/.virtual_lab/mcp_auth`, or the directory `VIRTUAL_LAB_MCP_AUTH_DIR` names, in a file for each server that only you can read, and are refreshed as they expire, so the next connection needs no sign-in. Anyone who can read the file can use the account, so treat it as you would a password. `sign_out_mcp("proto_hosted")`, or the server's URL, forgets a sign-in, so the next connection signs in again, as another account, say.

Where no browser can be opened, as over SSH, the sign-in page's address is shown at the terminal, to be opened anywhere, and the address the browser ends at, which may not load, is pasted back. Where there is neither a browser nor a terminal, as in a script run in the background, the connection fails and says so: connect once where there is one, and the sign-in kept is used from then on. The time you take to sign in, up to 10 minutes, is not counted against `start_timeout` or `timeout`.

#### Proto's tools in code

Code in a session can also call Proto's tools as a Python library, rather than through its server, by running in the interpreter Proto is installed in. Its tools run on your Modal account with `device="modal"`, which finds your Modal token in your home directory, or on Proto's servers with `device="proto"`, which needs an API key in `PROTO_API_KEY`, from https://proto.evodesign.org/settings/workspace/keys:

```python
with LocalSession(
    Path("results/session"),
    python=str(Path.home() / ".local/share/uv/tools/proto-tools/bin/python"),   # where uv tool install puts it
    forward_env=("PROTO_API_KEY",),
    software={"proto_tools": "Proto: run_esmfold, run_alphafold2, and over a hundred more run_* functions, "
                             "each taking an Input and a Config, such as ESMFoldConfig(device='proto')."},
) as session:
    hold_meeting(..., session=session)
```

Code run this way is not asked about first, as a call through a server is, so give a session Proto's library only where you would let its code spend on your accounts.

### Serving tools over MCP

`virtual-lab-mcp` serves tools to any MCP client: Claude Desktop, Claude Code, Cursor, another agent, or `connect_mcp`. It can serve Biomni's tool functions, the `run_code` tool, the database and literature tools above, and tools of your own, in any combination. `python -m virtual_lab.mcp_server` is the same command.

```bash
virtual-lab-mcp --biomni-tools --code-tool --directory results/served            # over stdio, for a client that starts it
virtual-lab-mcp --biomni-tools --biomni-module database --data-lake data/biomni_data --directory results/served
virtual-lab-mcp --database-tools --tool my_tools:TOOLS --transport http             # at http://127.0.0.1:8000/mcp
```

For Claude Desktop, add the server to `claude_desktop_config.json`, giving the full path to the command if Claude does not find it on its own:

```json
{
  "mcpServers": {
    "virtual-lab": {
      "command": "virtual-lab-mcp",
      "args": ["--biomni-tools", "--code-tool", "--directory", "/Users/me/lab/served"]
    }
  }
}
```

Biomni's `create_mcp_server` runs its functions in the server's own process, passes every parameter whose type it does not recognise as a string, and reports a function that fails as if it had returned its error. Here:

- **Biomni's functions run in the sandbox.** Each call runs in one session, a `DockerSession` in the `full` image unless `--image-target` says otherwise, with its working directory at `--directory`, where files the functions write are kept. `--session local` runs them on your machine instead, with no isolation, and says so. `--data-lake`, `--no-network`, and `--forward-env` are as for `session_executor`. Only the modules that import in the session are served, and those that do not are named in a warning; `--biomni-module` serves only the modules given.
- **Parameters are typed.** Biomni's types become JSON schema, so an `int` is an integer, a `List[str]` an array of strings, and a `pd.DataFrame` an array of rows. Arrays are made the numpy arrays, data frames, or tuples the function expects before it is called. A default the schema can hold is given as the default, and a parameter that takes a function, which cannot be sent, is left out.
- **The session is shared.** With `--code-tool`, `run_code` runs code in the same session, which keeps its variables from one call to the next. Whatever a Biomni function returned last is there as `_`, and a file it wrote can be read.
- **Arguments are checked.** Each call's arguments are checked against the tool's parameters before it runs, and every problem found is listed for the client, which can correct its call.
- **What a tool returns is sent back whole.** A dict, or a pydantic model, is sent as the structured result as well as text. A Biomni function that returns nothing gives back what it printed instead.
- **A failure is a failure.** A tool that raises an error is reported as failed, with the error, and for a Biomni function with the end of what it printed. The server carries on.
- **Calls run at the same time**, each in a thread of its own, except that calls to one session wait their turn.

`--tool` serves one of the database tools by name (`--tool uniprot_lookup`), or `module:attribute` for a `Tool`, a list of them, what `connect_mcp` returns, or a function, which is made a tool as `tool_from_function` makes it. The module is imported from the working directory. In Python, `serve_mcp(tools)` serves a list of tools, and `create_mcp_server(tools)` returns the server for you to run.

Over HTTP the server listens at `127.0.0.1`, which only this machine can reach, and refuses requests that name another host, so a web page cannot reach it by DNS rebinding. To listen anywhere else, such as `--host 0.0.0.0`, set a token of at least 16 characters in `VIRTUAL_LAB_MCP_TOKEN`, in the environment or in the `.env` file. Every request must then carry it as `Authorization: Bearer <token>`, which `connect_mcp` sends from the `headers` of its config. A token can be set on this machine too. `--env-file` reads another `.env` file before the server starts.

The server stops when its client does, or at Ctrl-C or `SIGTERM`, and stops its session's container as it goes. A Biomni function runs on arguments a model chose, and with the network on it can reach anything the container can, so serve them in a session you would let any of the server's clients use.

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
