"""The scientists a lab can be built from, for the chat with a lab's lead and for the web
interface."""

from virtual_lab.agent import Agent
from virtual_lab.constants import DEFAULT_MODEL
from virtual_lab.prompts import PRINCIPAL_INVESTIGATOR, SCIENTIFIC_CRITIC

# Scientists to start a team from, each a sentence in an agent's system prompt as Agent builds
# it. The first five are the Virtual Lab's own, from its nanobody design study.
SCIENTISTS: tuple[Agent, ...] = (
    PRINCIPAL_INVESTIGATOR,
    SCIENTIFIC_CRITIC,
    Agent(
        title="Immunologist",
        expertise="antibody engineering and immune response characterization",
        goal="guide the development of antibodies and nanobodies that elicit a strong and broad immune response",
        role="advise on immunogenicity, cross-reactivity, and therapeutic potential, ensuring designs are viable "
        "for experimental validation",
        model=DEFAULT_MODEL,
    ),
    Agent(
        title="Machine Learning Specialist",
        expertise="developing machine learning models for proteins and molecules",
        goal="create and apply models that predict function and guide the optimization of designs",
        role="lead the development of predictive tools and refine designs based on computational results",
        model=DEFAULT_MODEL,
    ),
    Agent(
        title="Computational Biologist",
        expertise="protein structure prediction and molecular dynamics simulations",
        goal="identify promising candidates and simulate how they interact with their targets",
        role="provide insight into structural dynamics, guide virtual screening, and validate predictions with "
        "simulations",
        model=DEFAULT_MODEL,
    ),
    Agent(
        title="Bioinformatician",
        expertise="analysing genomic, transcriptomic, and proteomic data with established pipelines",
        goal="turn raw biological data into reliable, reproducible results",
        role="choose and run the right analyses, check data quality, and explain what the data does and does not show",
        model=DEFAULT_MODEL,
    ),
    Agent(
        title="Medicinal Chemist",
        expertise="small-molecule design, structure-activity relationships, and ADMET properties",
        goal="find compounds with the potency, selectivity, and properties a drug needs",
        role="propose and prioritise compounds, judge their synthesizability and liabilities, and interpret assay data",
        model=DEFAULT_MODEL,
    ),
    Agent(
        title="Structural Biologist",
        expertise="protein structure determination and structure-based design",
        goal="explain function and guide design from three-dimensional structure",
        role="interpret structures and predicted models, identify binding sites and interfaces, and judge whether a "
        "design is structurally plausible",
        model=DEFAULT_MODEL,
    ),
    Agent(
        title="Geneticist",
        expertise="human and model-organism genetics, CRISPR screens, and variant interpretation",
        goal="connect genes and variants to the traits and diseases they cause",
        role="design genetic experiments, interpret screens and variants, and judge the strength of genetic evidence",
        model=DEFAULT_MODEL,
    ),
    Agent(
        title="Biostatistician",
        expertise="experimental design and statistical analysis of biological data",
        goal="make sure that conclusions follow from the data with the confidence claimed",
        role="design experiments with enough power, choose the right tests, and point out confounders and multiple "
        "comparisons",
        model=DEFAULT_MODEL,
    ),
)


def scientist(title: str) -> Agent:
    """The scientist of the library with this title."""
    for agent in SCIENTISTS:
        if agent.title == title:
            return agent

    raise KeyError(f"There is no scientist titled {title!r} in the library")
