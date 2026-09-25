"""Prompts for the language model agents and meetings."""

from typing import Iterable

from virtual_lab.agent import Agent
from virtual_lab.constants import DEFAULT_MODEL


PRINCIPAL_INVESTIGATOR = Agent(
    title="Principal Investigator",
    expertise="running a science research lab",
    goal="perform research in your area of expertise that maximizes the scientific impact of the work",
    role="lead a team of experts to solve an important scientific problem, make key decisions about the project direction based on team member input, and manage the project timeline and resources",
    model=DEFAULT_MODEL,
)

SCIENTIFIC_CRITIC = Agent(
    title="Scientific Critic",
    expertise="providing critical feedback for scientific research",
    goal="ensure that proposed research projects and implementations are rigorous, detailed, feasible, and scientifically sound",
    role="provide critical feedback to identify and correct all errors and demand that scientific answers that are maximally complete and detailed but simple and not overly complex",
    model=DEFAULT_MODEL,
)

SYNTHESIS_PROMPT = "synthesize the points raised by each team member, make decisions regarding the agenda based on team member input, and ask follow-up questions to gather more information and feedback about how to better address the agenda"

SUMMARY_PROMPT = "summarize the meeting in detail for future discussions, provide a specific recommendation regarding the agenda, and answer the agenda questions (if any) based on the discussion while strictly adhering to the agenda rules (if any)"

MERGE_PROMPT = "Please read the summaries of multiple separate meetings about the same agenda. Based on the summaries, provide a single answer that merges the best components of each individual answer. Please use the same format as the individual answers. Additionally, please explain what components of your answer came from each individual answer and why you chose to include them in your answer."

REWRITE_PROMPT = "This script needs to be improved. Please rewrite the script to make the following improvements without changing anything else."


def create_merge_prompt(
    agenda: str,
    agenda_questions: tuple[str, ...] = (),
    agenda_rules: tuple[str, ...] = (),
) -> str:
    """Creates a merge prompt for merging the best components of multiple separate meeting answers.

    :param agenda: The original agenda for the separate meetings.
    :param agenda_questions: The original agenda questions for the separate meetings.
    :param agenda_rules: The original agenda rules for the separate meetings.
    :return: The merge prompt.
    """
    return (
        f"{MERGE_PROMPT}\n\n"
        f"{format_agenda(agenda, intro='As a reference, here is the agenda from those meetings, which must be addressed here as well:')}"
        f"{format_agenda_questions(agenda_questions, intro='As a reference, here are the agenda questions from those meetings, which must be answered here as well:')}"
        f"{format_agenda_rules(agenda_rules, intro='As a reference, here are the agenda rules from those meetings, which must be followed here as well:')}"
    )


def summary_structure_prompt(has_agenda_questions: bool) -> str:
    """Formats the structure of a summary prompt.

    :param has_agenda_questions: Whether the summary prompt includes agenda questions.
    :return: The structure of a summary prompt.
    """
    if has_agenda_questions:
        agenda_questions_structure = [
            "### Answers",
            "For each agenda question, please provide the following:",
            "Answer: A specific answer to the question based on your recommendation above.",
            "Justification: A brief explanation of why you provided that answer.",
        ]
    else:
        agenda_questions_structure = []

    return "\n\n".join(
        [
            "### Agenda",
            "Restate the agenda in your own words.",
            "### Team Member Input",
            "Summarize all of the important points raised by each team member. This is to ensure that key details are preserved for future meetings.",
            "### Recommendation",
            "Provide your expert recommendation regarding the agenda. You should consider the input from each team member, but you must also use your expertise to make a final decision and choose one option among several that may have been discussed. This decision can conflict with the input of some team members as long as it is well justified. It is essential that you provide a clear, specific, and actionable recommendation. Please justify your recommendation as well.",
        ]
        + agenda_questions_structure
        + [
            "### Next Steps",
            "Outline the next steps that the team should take based on the discussion.",
        ]
    )


def format_prompt_list(prompts: Iterable[str]) -> str:
    """Formats prompts as a numbered list.

    :param prompts: The prompts.
    :return: The prompts formatted as a numbered list.
    """
    return f"{'\n\n'.join(f'{i + 1}. {prompt}' for i, prompt in enumerate(prompts))}"


def format_agenda(
    agenda: str, intro: str = "Here is the agenda for the meeting:"
) -> str:
    """Formats the agenda for the prompt.

    :param agenda: The agenda.
    :param intro: The introduction to the agenda.
    :return: The formatted agenda.
    """
    return f"{intro}\n\n{agenda}\n\n"


def format_agenda_questions(
    agenda_questions: tuple[str, ...],
    intro: str = "Here are the agenda questions that must be answered:",
) -> str:
    """Formats the agenda questions for the prompt as a numbered list.

    :param agenda_questions: The agenda questions.
    :param intro: The introduction to the agenda questions.
    :return: The formatted agenda questions.
    """
    return (
        f"{intro}\n\n{format_prompt_list(agenda_questions)}\n\n"
        if agenda_questions
        else ""
    )


def format_agenda_rules(
    agenda_rules: tuple[str, ...],
    intro: str = "Here are the agenda rules that must be followed:",
) -> str:
    """Formats the agenda rules for the prompt as a numbered list.

    :param agenda_rules: The agenda rules.
    :param intro: The introduction to the agenda rules.
    :return: The formatted agenda rules.
    """
    return f"{intro}\n\n{format_prompt_list(agenda_rules)}\n\n" if agenda_rules else ""


def format_references(
    references: tuple[str, ...], reference_type: str, intro: str
) -> str:
    """Formats references (e.g., contexts, summaries) for the prompt.

    :param references: The references.
    :param reference_type: The type of the references (e.g., "context", "summary").
    :param intro: The introduction to the references.
    :return: The formatted references.
    """
    if not references:
        return ""

    formatted_references = [
        f"[begin {reference_type} {reference_index + 1}]\n\n{reference}\n\n[end {reference_type} {reference_index + 1}]"
        for reference_index, reference in enumerate(references)
    ]

    return f"{intro}\n\n{'\n\n'.join(formatted_references)}\n\n"


# Team meeting prompts
def team_meeting_start_prompt(
    team_lead: Agent,
    team_members: tuple[Agent, ...],
    agenda: str,
    agenda_questions: tuple[str, ...] = (),
    agenda_rules: tuple[str, ...] = (),
    summaries: tuple[str, ...] = (),
    contexts: tuple[str, ...] = (),
    num_rounds: int = 1,
) -> str:
    """Generates the start prompt for a tean meeting.

    :param team_lead: The team lead.
    :param team_members: The team members.
    :param agenda: The agenda for the meeting.
    :param agenda_questions: The agenda questions to answer by the end of the meeting.
    :param agenda_rules: The rules for the agenda.
    :param summaries: The summaries of previous meetings.
    :param contexts: The contexts for the meeting.
    :param num_rounds: The number of rounds of discussion.
    :return: The start prompt for the tean meeting.
    """
    return (
        f"This is the beginning of a team meeting to discuss your research project. "
        f"This is a meeting with the team lead, {team_lead.title}, and the following team members: "
        f"{', '.join(team_member.title for team_member in team_members)}.\n\n"
        f"{format_references(contexts, reference_type='context', intro='Here is context for this meeting:')}"
        f"{format_references(summaries, reference_type='summary', intro='Here are summaries of the previous meetings:')}"
        f"{format_agenda(agenda)}"
        f"{format_agenda_questions(agenda_questions)}"
        f"{format_agenda_rules(agenda_rules)}"
        f"{team_lead} will convene the meeting. "
        f"Then, each team member will provide their thoughts on the discussion one-by-one in the order above. "
        f"After all team members have given their input, {team_lead} will {SYNTHESIS_PROMPT}. "
        f"This will continue for {num_rounds} rounds. Once the discussion is complete, {team_lead} will {SUMMARY_PROMPT}."
    )


def team_meeting_team_lead_initial_prompt(team_lead: Agent) -> str:
    """Generates the initial prompt for the team lead in a team meeting.

    :param team_lead: The team lead.
    :return: The initial prompt for the team lead.
    """
    return f"{team_lead}, please provide your initial thoughts on the agenda as well as any questions you have to guide the discussion among the team members."


def team_meeting_team_member_prompt(
    team_member: Agent, round_num: int, num_rounds: int
) -> str:
    """Generates the prompt for a team member in a team meeting.

    :param team_member: The team member.
    :param round_num: The current round number.
    :param num_rounds: The total number of rounds.
    :return: The prompt for the team member.
    """
    return (
        f"{team_member}, please provide your thoughts on the discussion (round {round_num} of {num_rounds}). "
        f'If you do not have anything new or relevant to add, you may say "pass". '
        f"Remember that you can and should (politely) disagree with other team members if you have a different perspective."
    )


def team_meeting_team_lead_intermediate_prompt(
    team_lead: Agent, round_num: int, num_rounds: int
) -> str:
    """Generates the intermediate prompt for the team lead in a team meeting at the end of a round of discussion.

    :param team_lead: The team lead.
    :param round_num: The current round number.
    :param num_rounds: The total number of rounds.
    :return: The intermediate prompt for the team lead.
    """
    return f"This concludes round {round_num} of {num_rounds} of discussion. {team_lead}, please {SYNTHESIS_PROMPT}."


def team_meeting_team_lead_final_prompt(
    team_lead: Agent,
    agenda: str,
    agenda_questions: tuple[str, ...] = (),
    agenda_rules: tuple[str, ...] = (),
) -> str:
    """Generates the final prompt for the team lead in a team meeting to summarize the discussion.

    :param team_lead: The team lead.
    :param agenda: The agenda for the meeting.
    :param agenda_questions: The agenda questions to answer by the end of the meeting.
    :param agenda_rules: The rules for the agenda.
    :return: The final prompt for the team lead.
    """
    return (
        f"{team_lead}, please {SUMMARY_PROMPT}.\n\n"
        f"{format_agenda(agenda, intro='As a reminder, here is the agenda for the meeting:')}"
        f"{format_agenda_questions(agenda_questions, intro='As a reminder, here are the agenda questions that must be answered:')}"
        f"{format_agenda_rules(agenda_rules, intro='As a reminder, here are the agenda rules that must be followed:')}"
        f"Your summary should take the following form.\n\n"
        f"{summary_structure_prompt(has_agenda_questions=len(agenda_questions) > 0)}"
    )


# Individual meeting prompts
def individual_meeting_start_prompt(
    team_member: Agent,
    agenda: str,
    agenda_questions: tuple[str, ...] = (),
    agenda_rules: tuple[str, ...] = (),
    summaries: tuple[str, ...] = (),
    contexts: tuple[str, ...] = (),
) -> str:
    """Generates the start prompt for an individual meeting.

    :param team_member: The team member.
    :param agenda: The agenda for the meeting.
    :param agenda_questions: The agenda questions to answer by the end of the meeting.
    :param agenda_rules: The rules for the agenda.
    :param summaries: The summaries of previous meetings.
    :param contexts: The contexts for the meeting.
    :return: The start prompt for the individual meeting.
    """
    return (
        f"This is the beginning of an individual meeting with {team_member} to discuss your research project.\n\n"
        f"{format_references(contexts, reference_type='context', intro='Here is context for this meeting:')}"
        f"{format_references(summaries, reference_type='summary', intro='Here are summaries of the previous meetings:')}"
        f"{format_agenda(agenda)}"
        f"{format_agenda_questions(agenda_questions)}"
        f"{format_agenda_rules(agenda_rules)}"
        f"{team_member}, please provide your response to the agenda."
    )


def individual_meeting_critic_prompt(
    critic: Agent,
    agent: Agent,
) -> str:
    """Generates the intermediate prompt for the critic in an individual meeting.

    :param critic: The critic.
    :param agent: The agent that the critic is criticizing.
    """
    return (
        f"{critic.title}, please critique {agent.title}'s most recent answer. "
        "In your critique, suggest improvements that directly address the agenda and any agenda questions. "
        "Prioritize simple solutions over unnecessarily complex ones, but demand more detail where detail is lacking. "
        "Additionally, validate whether the answer strictly adheres to the agenda and any agenda questions and provide corrective feedback if it does not. "
        "Only provide feedback; do not implement the answer yourself."
    )


def individual_meeting_agent_prompt(
    critic: Agent,
    agent: Agent,
) -> str:
    """Generates the intermediate prompt for the agent in an individual meeting.

    :param critic: The critic.
    :param agent: The agent.
    """
    return (
        f"{agent.title}, please modify your answer to address {critic.title}'s most recent feedback. "
        "Remember that your ultimate goal is to make improvements that better address the agenda."
    )


def structured_output_prompt(agent: Agent) -> str:
    """Generates the prompt asking an agent to restate the meeting's conclusions as data.

    :param agent: The agent who closed the meeting and will restate its conclusions.
    """
    return (
        f"{agent.title}, please restate the conclusions of this meeting in the required structured "
        f"format. Report only what the meeting decided; do not introduce new content, and do not "
        f"soften or omit a conclusion to make it fit. If the meeting genuinely did not settle "
        f"something the format requires, choose the option best supported by the discussion above."
    )


CODING_RULES = (
    "Your code must be self-contained (with appropriate imports) and complete.",
    "Your code may not include any undefined or unimplemented variables or functions.",
    "Your code may not include any pseudocode; it must be fully functioning code.",
    "Your code may not include any hard-coded examples.",
    "If your code needs user-provided values, write code to parse those values from the command line.",
    "Your code must be high quality, well-engineered, efficient, and well-documented (including docstrings, comments, and Python type hints if using Python).",
)
