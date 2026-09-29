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
