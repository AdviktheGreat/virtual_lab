"""Running a project to its goal, with its principal investigator deciding every step.

hold_meeting runs one meeting, and Project keeps many within one budget, but someone still has to
decide what each one is about. run_project hands that to the team lead, the principal
investigator. Once a team is chosen and a plan made, every round the principal investigator reads
what the project has done so far and decides one next step: a team meeting, an individual
meeting, code to write and run, a change to the team, or an end to the project with its answer.
The critic has to agree that the answer meets the goal before the project ends with it.

A run stops when the project finishes, when its budget or its rounds run out, when the plan
stops moving forward, or when the approve hook says to. Every step is a step of the Project, so a
run that stopped is carried on by running it again on the same directory: the steps it took are
read back for nothing, and it goes on from the round it stopped in. Nothing a step is asked
depends on the budget or the limits on rounds, so raising them to carry on changes none of the
steps already taken.

What the work finds is kept in a LabMemory, as findings, and each step is given the findings it
needs rather than the summary of every step before it: those the principal investigator names
when it decides the step, by default, or those that best match the step's agenda.
"""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field

from virtual_lab.agent import Agent
from virtual_lab.artifacts import CodeArtifacts
from virtual_lab.constants import MEMORY_FILE_NAME, REPORT_FILE_NAME, REPORT_MARKDOWN_FILE_NAME, RESEARCH_LOG_FILE_NAME
from virtual_lab.events import MeetingEvent, OnProjectEvent, ProjectEvent, ProjectEventKind
from virtual_lab.execution import Executor
from virtual_lab.memory import Findings, LabMemory, MemoryEntry
from virtual_lab.project import REPAIR_OPTIONS, Project, ProjectBudgetExceededError
from virtual_lab.prompts import CODING_RULES, PRINCIPAL_INVESTIGATOR, SCIENTIFIC_CRITIC
from virtual_lab.provenance import describe_agent
from virtual_lab.run_meeting import MeetingResult
from virtual_lab.schemas import AgentSpec, TeamRoster, normalize_title
from virtual_lab.utils import write_atomically

Action = Literal["team_meeting", "individual_meeting", "write_code", "change_team", "finish"]

Status = Literal["finished", "out_of_budget", "out_of_rounds", "stalled", "stopped"]

Memory = Literal["pick", "bm25", "summaries"]

# Code is run by the project as it is, with no one there to give it anything
PROJECT_CODING_RULES = tuple(
    rule for rule in CODING_RULES if not rule.startswith("If your code needs user-provided values")
) + (
    "Your code will be run as it is, with no arguments and no one to answer it, so it must not "
    "read from the command line or wait for input.",
    "Your code must print what it finds, since what it prints is all the project learns of it.",
)


class PlanTask(BaseModel):
    """A task in a project's plan, and how far it has got."""

    task: str = Field(
        description="A piece of work the project needs, stated so that whether it has been done can be judged."
    )
    status: Literal["to do", "in progress", "done", "dropped"] = Field(
        description='"done" only once the work so far shows it done, and "dropped" if the project no longer needs it.'
    )


class ResearchPlan(BaseModel):
    """The plan a project starts with."""

    tasks: list[str] = Field(
        description="The work the project needs to reach its goal, in the order it should be done, each a task "
        "stated so that whether it has been done can be judged."
    )


class NextStep(BaseModel):
    """What the principal investigator decided the project does next.

    Every field is required, as the API's structured output requires; the fields an action does
    not use are left empty.
    """

    plan: list[PlanTask] = Field(
        description="The whole plan as it now stands, revised in light of what the project has learned: every "
        "task, with its status, including the ones done and dropped."
    )
    progress: str = Field(description="What the work so far has established towards the goal, and what it has not.")
    action: Action = Field(
        description="team_meeting, individual_meeting, write_code, change_team, or finish, as the agenda describes "
        "each."
    )
    rationale: str = Field(description="Why this is the most useful next step.")
    participants: list[str] = Field(
        description="The exact titles of who takes part: one or more for a team_meeting, exactly one for an "
        "individual_meeting or write_code, and none for change_team or finish."
    )
    agenda: str = Field(
        description="For a meeting or write_code, what to work on, stated completely enough to be worked on with "
        "nothing else to go on. Empty for change_team or finish."
    )
    agenda_questions: list[str] = Field(
        description="For a meeting or write_code, specific questions the work must answer. May be empty."
    )
    findings: list[str] = Field(
        description="The ids of the listed findings the step needs in full: for a meeting or write_code, those it "
        "builds on, and for finish, those the answer rests on. Empty if none are listed, or if the agenda says "
        "findings are chosen another way."
    )
    add_members: list[AgentSpec] = Field(description="For change_team, the scientists to bring onto the team.")
    remove_members: list[str] = Field(description="For change_team, the exact titles of the members to let go.")
    answer: str = Field(
        description="For finish, the project's answer to its goal, complete and self-contained, with the evidence "
        "the work found for it. Empty otherwise."
    )


class Review(BaseModel):
    """What the critic made of the answer the principal investigator proposed to end with."""

    goal_met: bool = Field(
        description="Whether the answer meets the project's goal, as the work done shows it: false if anything the "
        "goal needs is missing, unsupported by the work, or wrong."
    )
    objections: list[str] = Field(
        description="What is missing, unsupported, or wrong, each specific enough to act on. Empty only if the goal "
        "is met."
    )


@dataclass
class ProjectRound:
    """One round of a project run to its goal, as its research log records it.

    :param number: The round, counting from one.
    :param proposed: The principal investigator's decision, as NextStep.model_dump gives it.
    :param approved: The decision as carried out, after the approve hook, or None if the hook
        stopped the project instead.
    :param outcome: "done" if the step was taken, "invalid" if the decision could not be carried
        out, "objected" if the critic did not accept the answer, "finished" if it did, "stopped"
        if the approve hook stopped the project, "out_of_budget" if the budget ran out during
        the step, or "running" if the process stopped during it.
    :param note: Why a decision could not be carried out, or what the critic objected to.
    :param steps: The project's steps the round took, by name.
    :param team: The team's titles after the round, the principal investigator and critic aside.
    :param tasks_done: How many of the plan's tasks were done after the round.
    :param findings: The findings the round's step was given, by id.
    :param found: The findings the round's step made, by id.
    """

    number: int
    proposed: dict[str, Any]
    approved: dict[str, Any] | None
    outcome: str
    note: str = ""
    steps: list[str] = field(default_factory=list)
    team: list[str] = field(default_factory=list)
    tasks_done: int = 0
    findings: list[str] = field(default_factory=list)
    found: list[str] = field(default_factory=list)


@dataclass
class ProjectReport:
    """How a project run to its goal ended.

    :param goal: The project's goal.
    :param status: "finished" if the critic accepted an answer, "out_of_budget",
        "out_of_rounds", "stalled" if the plan stopped moving forward, or "stopped" if the
        approve hook stopped it.
    :param reason: Why it ended, in a sentence.
    :param answer: The answer the critic accepted, if it finished.
    :param proposed_answer: The last answer the critic did not accept, if it did not finish.
    :param objections: What the critic last objected to.
    :param team_lead: The principal investigator, as describe_agent gives it.
    :param critic: The critic, as describe_agent gives it.
    :param team: The team at the end, as describe_agent gives each member.
    :param team_changes: Every change to the team: the round, who was added and removed, and why.
    :param plan: The plan at the end, as PlanTask.model_dump gives each task.
    :param findings: Every finding the project made, as its LabMemory keeps them.
    :param rounds: Every round.
    :param spent: What the project has spent in USD, over every run of it, or None if unknown.
    :param max_cost: The project's limit, if it has one.
    """

    goal: str
    status: Status
    reason: str
    answer: str | None
    proposed_answer: str | None
    objections: list[str]
    team_lead: dict[str, str]
    critic: dict[str, str]
    team: list[dict[str, str]]
    team_changes: list[dict[str, Any]]
    plan: list[dict[str, str]]
    findings: list[dict[str, Any]]
    rounds: list[ProjectRound]
    spent: float | None
    max_cost: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        """The report as a document for a person to read."""
        lines = [
            "# Project report",
            "",
            f"**Goal:** {self.goal}",
            "",
            f"**Status:** {self.status}. {self.reason}",
            "",
        ]
        if self.answer is not None:
            lines += ["## Answer", "", self.answer, ""]
        elif self.proposed_answer is not None:
            lines += ["## Last answer proposed, which the critic did not accept", "", self.proposed_answer, ""]
        if self.objections:
            lines += ["## The critic's objections", "", *(f"- {objection}" for objection in self.objections), ""]

        lines += ["## Team", "", f"- {self.team_lead['title']} (team lead)", f"- {self.critic['title']} (critic)"]
        lines += [f"- {member['title']}: {member['expertise']}" for member in self.team]
        lines += [""]

        marks = {"done": "[x]", "dropped": "[-]", "in progress": "[~]", "to do": "[ ]"}
        lines += ["## Plan", "", *(f"- {marks[task['status']]} {task['task']}" for task in self.plan), ""]

        if self.findings:
            lines += ["## Findings", "", *(f"- [{finding['id']}] {' '.join(finding['claim'].split())}" for finding in self.findings), ""]

        lines += ["## Rounds", "", "| Round | Decision | Outcome |", "| --- | --- | --- |"]
        for round_ in self.rounds:
            decision = round_.approved or round_.proposed
            what = decision["action"]
            if decision["participants"]:
                what += f" ({', '.join(decision['participants'])})"
            if decision["agenda"]:
                what += f": {decision['agenda']}"
            outcome = round_.outcome + (f": {round_.note}" if round_.note else "")
            lines.append(f"| {round_.number} | {table_cell(what)} | {table_cell(outcome)} |")
        lines += [""]

        cost = "unknown" if self.spent is None else f"${self.spent:.4f}"
        limit = "" if self.max_cost is None else f" of a limit of ${self.max_cost:.4f}"
        lines += ["## Cost", "", f"{cost}{limit}", ""]

        return "\n".join(lines)


def table_cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def run_project(
    project: Project,
    team_lead: Agent = PRINCIPAL_INVESTIGATOR,
    critic: Agent = SCIENTIFIC_CRITIC,
    team: tuple[Agent, ...] | None = None,
    max_team_size: int = 3,
    max_rounds: int = 10,
    max_stalled_rounds: int = 3,
    meeting_rounds: int = 1,
    executor: Executor | None = None,
    approve: Callable[[int, NextStep], NextStep | None] | None = None,
    repair_options: dict[str, Any] | None = None,
    memory: Memory = "pick",
    findings_per_step: int = 8,
    on_event: OnProjectEvent | None = None,
) -> ProjectReport:
    """Runs a project to its goal, with its team lead deciding each step, and reports how it ended.

    The team lead chooses a team, unless one is given, and the team makes a plan. Then every
    round the team lead decides one next step, as a NextStep, with the plan restated:

    - team_meeting: a discussion the team lead leads, with the team members and critic it names.
    - individual_meeting: one team member, or the team lead, works on an agenda, with the critic
      reviewing for meeting_rounds rounds.
    - write_code: one of them writes code, which is run with executor and repaired if it fails.
      Offered only with an executor.
    - change_team: members are brought onto the team or let go.
    - finish: the team lead gives the project's answer, which the critic reviews. The project
      ends only if the critic agrees the answer meets the goal; otherwise the critic's objections
      are what the next round has to go on.

    Every meeting is told the goal, the team, and the plan. What the work finds is kept as
    findings: each meeting restates what it established as Findings, in one more request, and
    the author of code that was run says what the run established, in a short meeting of its
    own. What else a step is given depends on memory:

    - "pick": the team lead is shown every finding by its id and claim, with what each step
      found, and names in each decision the findings the step needs, which it is given in full.
      This is how Biomni's agent chooses the resources a task needs, and costs no more requests.
    - "bm25": each step is given the findings that best match its agenda and questions, or a
      proposed answer, by BM25, at most findings_per_step of them.
    - "summaries": no findings are kept, and each step is given the summary of every step before
      it, as the Virtual Lab's meetings are. Context grows with every step.

    A decision that cannot be carried out, such as one naming someone not on the team, is
    recorded, and the team lead is told why in the next round. The research log,
    research_log.json, is saved after every round, and the report, report.json and report.md, at
    the end, in the project's directory. A run that fails with an error other than running out
    of budget leaves no report, and the log says what the error was.

    Nothing a step is asked depends on the project's budget, max_rounds, or max_stalled_rounds,
    so a project that stopped for one of them is carried on by raising it and running it again:
    the steps already taken are read back from disk for nothing. The team lead is therefore not
    told how many rounds or how much money is left.

    :param project: The project, which holds the goal, the budget, and every step.
    :param team_lead: Who leads the project and decides each step.
    :param critic: Who critiques the work, and has to agree before the project ends.
    :param team: The team, the team lead and critic aside, or None for the team lead to choose
        one of at most max_team_size, with team_lead's model.
    :param max_team_size: The most members the team lead may have on the team.
    :param max_rounds: The most rounds the project may take, each one decision and its step.
    :param max_stalled_rounds: How many rounds in a row may pass without one more of the plan's
        tasks done before the project is stopped as stalled.
    :param meeting_rounds: The rounds of discussion in each meeting, and of critique in each
        individual meeting.
    :param executor: What to run code with, for write_code.
    :param approve: Called with the round and the team lead's decision before it is carried out.
        It returns the decision to carry out, which may be changed, or None to stop the project.
        Every decision carried out, approved or taken with no hook, is recorded in the research
        log and is not asked about again when the project is carried on, however far an earlier
        run got; one the hook stopped is asked about again.
    :param repair_options: Options for run_with_repair, as Project.repair takes them, for code.
    :param memory: How each step is given what the work before it found: "pick", "bm25", or
        "summaries", as above.
    :param findings_per_step: The most findings a step is given with "bm25".
    :param on_event: Called with a ProjectEvent when the team is chosen or changed, the plan is
        made, each step is decided, code is run, each round ends, and the project ends, and with
        a MeetingEvent for everything that happens in each of its meetings, as hold_meeting's
        on_event is, so that the project can be followed live. A meeting read back from disk is
        told of by one "read_back" event. The project's own on_event, if it was given one as a
        meeting option, is still called first. An exception it raises stops the project the way
        any other error does, except from the project's "finished" event, once the report is
        saved, which is warned about.
    :raises ProjectStateError: If the project's directory holds steps taken with other inputs:
        another team lead, critic, team, max_team_size, meeting_rounds, or memory, a
        findings_per_step that changes which findings a step is given, or an executor or a
        session where there was none, or none where there was one.
    :raises CostUnknownError: If the project has a limit and a cost cannot be worked out.
    :return: How the project ended. Running out of budget ends it with a report, not an error.
    """
    for name, value, least in (
        ("max_team_size", max_team_size, 0),
        ("max_rounds", max_rounds, 1),
        ("max_stalled_rounds", max_stalled_rounds, 1),
        ("meeting_rounds", meeting_rounds, 0),
        ("findings_per_step", findings_per_step, 1),
    ):
        if value < least:
            raise ValueError(f"{name} must be at least {least}, not {value}")

    if memory not in get_args(Memory):
        raise ValueError(f'memory must be "pick", "bm25", or "summaries", not {memory!r}')

    if normalize_title(team_lead.title) == normalize_title(critic.title):
        raise ValueError("The team lead and the critic must have different titles")

    if team is not None:
        check_titles_are_free(team, taken=(team_lead, critic))

    if unknown := sorted(set(repair_options or {}) - REPAIR_OPTIONS):
        raise TypeError(f"run_with_repair takes no option {', '.join(unknown)} here")

    return ProjectRun(
        project=project,
        team_lead=team_lead,
        critic=critic,
        team=team,
        max_team_size=max_team_size,
        max_rounds=max_rounds,
        max_stalled_rounds=max_stalled_rounds,
        meeting_rounds=meeting_rounds,
        executor=executor,
        approve=approve,
        repair_options=dict(repair_options or {}),
        memory=memory,
        findings_per_step=findings_per_step,
        on_event=on_event,
    ).run()


def check_titles_are_free(members: tuple[Agent, ...], taken: tuple[Agent, ...]) -> None:
    """Refuses a team whose titles repeat, or are the team lead's or critic's."""
    titles = [normalize_title(member.title) for member in members]
    if not all(titles):
        raise ValueError("Every team member needs a title")
    if len(set(titles)) != len(titles):
        raise ValueError("Team members must have different titles")
    if clashes := sorted(member.title for member in members if normalize_title(member.title) in {normalize_title(agent.title) for agent in taken}):
        raise ValueError(f"A team member cannot have the team lead's or critic's title: {', '.join(clashes)}")


def article(word: str) -> str:
    return "An" if word[0] in "aeiou" else "A"


def find_agent(title: str, agents: list[Agent]) -> Agent | None:
    wanted = normalize_title(title)
    return next((agent for agent in agents if normalize_title(agent.title) == wanted), None)


def describe_member(agent: Agent) -> str:
    return f"{agent.title}, with expertise in {agent.expertise}, whose role is to {agent.role}"


@dataclass
class Event:
    """Something a project did, as the steps after it are told of it.

    :param what: What it was.
    :param found: What came of it.
    :param shown: Whether the team is told of it, as well as the team lead.
    :param work: Whether it was work, a meeting or code, rather than something that happened to
        the project, such as a change to the team.
    :param finding_ids: The findings it made, by id.
    """

    what: str
    found: str
    shown: bool
    work: bool = False
    finding_ids: tuple[str, ...] = ()


class ProjectRun:
    """One run of run_project: the team, the plan, and what has been done, round by round."""

    def __init__(
        self,
        project: Project,
        team_lead: Agent,
        critic: Agent,
        team: tuple[Agent, ...] | None,
        max_team_size: int,
        max_rounds: int,
        max_stalled_rounds: int,
        meeting_rounds: int,
        executor: Executor | None,
        approve: Callable[[int, NextStep], NextStep | None] | None,
        repair_options: dict[str, Any],
        memory: Memory = "pick",
        findings_per_step: int = 8,
        on_event: OnProjectEvent | None = None,
    ) -> None:
        self.project = project
        self.team_lead = team_lead
        self.critic = critic
        self.given_team = team
        self.team: list[Agent] = list(team or ())
        self.max_team_size = max_team_size
        self.max_rounds = max_rounds
        self.max_stalled_rounds = max_stalled_rounds
        self.meeting_rounds = meeting_rounds
        self.executor = executor
        self.approve = approve
        self.repair_options = repair_options
        self.memory_mode = memory
        self.findings_per_step = findings_per_step
        self.on_event = on_event
        # Built again on every run from the steps, which are read back, so it is what they found
        self.memory = LabMemory()

        self.plan: list[PlanTask] = []
        self.rounds: list[ProjectRound] = []
        self.team_changes: list[dict[str, Any]] = []
        self.history: list[Event] = []
        self.answer: str | None = None
        self.proposed_answer: str | None = None
        self.objections: list[str] = []

        # Every decision carried out, by round, as proposed and as carried out. A step taken on one
        # is read back only if it is asked for the same way, so these are kept however far a
        # later run gets, and are never asked about again.
        log = project.save_dir / RESEARCH_LOG_FILE_NAME
        saved = json.loads(log.read_text(encoding="utf-8")) if log.is_file() else {}
        self.approvals: dict[int, dict[str, Any]] = {int(number): entry for number, entry in saved.get("approvals", {}).items()}

    # Telling the caller

    # Positional only, so that the round's record can be told of in data under its own name
    def emit(self, kind: ProjectEventKind, round: int | None = None, text: str = "", /, **data: Any) -> None:
        if self.on_event is not None:
            self.on_event(ProjectEvent(kind=kind, round=round, text=text, data=data))

    def meeting(self, meeting_type: Literal["team", "individual"], agenda: str, **options: Any) -> MeetingResult:
        """Holds a meeting for the project, as Project.meeting does, telling on_event of what
        happens in it as well as the project's own on_event."""
        if self.on_event is not None:
            own = self.project.meeting_options.get("on_event")
            on_event = self.on_event

            def tell(event: MeetingEvent) -> None:
                if own is not None:
                    own(event)
                on_event(event)

            options["on_event"] = tell

        return self.project.meeting(meeting_type, agenda, **options)

    def run(self) -> ProjectReport:
        # A report left from an earlier run would say how a run that is no longer the last ended
        for name in (REPORT_FILE_NAME, REPORT_MARKDOWN_FILE_NAME):
            (self.project.save_dir / name).unlink(missing_ok=True)

        number = 0
        try:
            if self.given_team is None:
                self.choose_team()
            else:
                self.record_team_change(0, added=self.team, removed=[], why="The team was given.")
            self.make_plan()

            best = stalled = 0
            for number in range(1, self.max_rounds + 1):
                ended = self.play_round(number)
                self.save_log()
                if ended is not None:
                    return self.finish(*ended)

                done = self.tasks_done()
                if done > best:
                    best, stalled = done, 0
                else:
                    stalled += 1
                if stalled >= self.max_stalled_rounds:
                    return self.finish(
                        "stalled",
                        f"No more of the plan's tasks were done in the last {stalled} round{'s' if stalled > 1 else ''}.",
                    )

            return self.finish("out_of_rounds", f"The project took all {self.max_rounds} of its rounds without finishing.")
        except ProjectBudgetExceededError as error:
            try:
                if self.rounds and self.rounds[-1].outcome == "running":
                    self.rounds[-1].outcome = "out_of_budget"
                    self.finish_round(self.rounds[-1])
                when = f"in round {number}" if number else "before its first round"
                return self.finish("out_of_budget", f"The project ran out of budget {when}: {error}.")
            except BaseException as failure:
                self.save_log("failed", failure)
                raise
        except BaseException as error:
            self.save_log("failed", error)
            raise

    # Before the first round

    def choose_team(self) -> None:
        taken = f"Do not include yourself or the {self.critic.title}, who are on the project already."
        result = self.meeting(
            "individual",
            f"You are leading a research project with this goal:\n\n{self.project.goal}\n\nChoose the team of "
            f"scientists you need for it, at most {self.max_team_size}, each with the expertise the goal calls "
            f"for and a role in the project. {taken}",
            name="team",
            team_member=self.team_lead,
            critic=self.critic,
            num_rounds=self.meeting_rounds,
            output_schema=TeamRoster,
        )
        assert isinstance(result.output, TeamRoster)
        chosen = list(result.output.to_agents(model=self.team_lead.model))

        notes = []
        if untitled := [agent for agent in chosen if not normalize_title(agent.title)]:
            notes.append(f"{len(untitled)} left out, as without a title")
        free: list[Agent] = []
        clashing = []
        for agent in chosen:
            if agent in untitled:
                continue
            if find_agent(agent.title, [self.team_lead, self.critic, *free]) is None:
                free.append(agent)
            else:
                clashing.append(agent.title)
        if clashing:
            notes.append(f"{', '.join(clashing)} left out, as already on the project")
        if len(free) > self.max_team_size:
            notes.append(f"{', '.join(agent.title for agent in free[self.max_team_size :])} left out, as over the limit")
            free = free[: self.max_team_size]

        self.team = free
        why = "Chosen by the team lead." + (f" {'; '.join(notes)}." if notes else "")
        self.record_team_change(0, added=self.team, removed=[], why=why)

    def make_plan(self) -> None:
        agenda = (
            "Make the plan for this project: the work it needs to reach its goal, as a list of tasks in the order "
            "they should be done, each stated so that whether it has been done can be judged. Plan only work this "
            "team can do in meetings and, where it helps, in code, and keep the plan to what the goal needs."
        )
        if self.team:
            result = self.meeting(
                "team",
                agenda,
                name="plan",
                team_lead=self.team_lead,
                team_members=(*self.team, self.critic),
                contexts=(self.brief(),),
                num_rounds=self.meeting_rounds,
                output_schema=ResearchPlan,
            )
        else:
            result = self.meeting(
                "individual",
                agenda,
                name="plan",
                team_member=self.team_lead,
                critic=self.critic,
                contexts=(self.brief(),),
                num_rounds=self.meeting_rounds,
                output_schema=ResearchPlan,
            )
        assert isinstance(result.output, ResearchPlan)
        self.plan = [PlanTask(task=task, status="to do") for task in result.output.tasks]
        self.emit("plan", plan=[task.model_dump(mode="json") for task in self.plan])
        self.history.append(Event("The plan the team made", result.summary, shown=True, work=True))

    # Each round

    def play_round(self, number: int) -> tuple[Status, str] | None:
        """Decides a step and takes it.

        :return: How the project ended, if it ended in this round.
        """
        proposed = self.decide(number)
        self.emit("decided", number, proposed=proposed.model_dump(mode="json"))
        approved = self.approval(number, proposed)
        round_ = ProjectRound(
            number=number,
            proposed=proposed.model_dump(mode="json"),
            approved=approved.model_dump(mode="json") if approved is not None else None,
            outcome="running",
        )
        self.rounds.append(round_)
        # Saved before the step is taken, so that a step taken on an approval is never carried on
        # with another one, should the process stop before the round ends
        self.save_log()

        if approved is None:
            round_.outcome = "stopped"
            self.finish_round(round_)
            return "stopped", f"The approve hook stopped the project in round {number}."

        if (problem := self.check(approved)) is not None:
            round_.outcome, round_.note = "invalid", problem
            self.history.append(Event(f"Round {number}: your decision could not be carried out", problem, shown=False))
            self.finish_round(round_)
            return None

        self.plan = list(approved.plan)
        ended = self.take(number, approved, round_)
        if round_.outcome == "running":
            round_.outcome = "done"
        self.finish_round(round_)

        return ended

    def decide(self, number: int) -> NextStep:
        actions = [
            "- team_meeting: a discussion you lead with the participants you name, from the team and the "
            f"{self.critic.title}, on an agenda you set.",
            "- individual_meeting: one participant, you or a member of the team, works on an agenda you set, with "
            f"the {self.critic.title} critiquing the work.",
        ]
        if self.executor is not None:
            actions.append(
                "- write_code: one participant, you or a member of the team, writes code for an agenda you set, "
                "which is then run, and repaired if it fails; what it printed is reported back."
            )
        actions += [
            f"- change_team: bring scientists onto the team or let members go, keeping it to at most "
            f"{self.max_team_size}.",
            f"- finish: end the project with its answer. The {self.critic.title} reviews the answer against the "
            "goal and the work done, and the project ends only if it agrees the goal is met.",
        ]
        parts = [
            f"This is round {number} of the project. Decide the single most useful next step towards the goal, "
            "given the work done so far, as given above. The steps you can take:",
            "\n".join(actions),
        ]
        if self.project.session is not None:
            parts.append("Every meeting has a running session that code can be run in as it goes.")
        if self.memory_mode == "pick" and len(self.memory):
            parts.append(
                "The findings the project has made are listed above by id and claim. In findings, name the ones "
                "the step needs: a meeting or write_code is given those in full, and no others, and for finish, "
                f"name the ones the answer rests on, which the {self.critic.title} is given in full."
            )
        elif self.memory_mode == "pick":
            parts.append("The project has made no findings yet, so leave findings empty.")
        elif self.memory_mode == "bm25":
            parts.append(
                "Each step is given the findings whose words best match its agenda and questions, so state the "
                "agenda in the terms of the findings it builds on. Leave findings empty."
            )
        parts.append(
            "Restate the whole plan with each task's status, revised in light of what has been learned: mark a "
            "task done only when the work so far shows it done, and add or drop tasks as the work requires. "
            "Finish as soon as the work supports an answer that meets the goal, since every step costs time and "
            "money; do not finish before it does. Name participants by their exact titles. Leave the fields your "
            "step does not use empty."
        )
        agenda = "\n\n".join(parts)
        result = self.meeting(
            "individual",
            agenda,
            name=f"round_{number:02d}_decision",
            team_member=self.team_lead,
            summaries=tuple(self.decision_summaries()),
            contexts=self.contexts(),
            num_rounds=0,
            output_schema=NextStep,
        )
        assert isinstance(result.output, NextStep)

        return result.output

    def approval(self, number: int, proposed: NextStep) -> NextStep | None:
        """The decision to carry out: the one approved before, if this one was, or the hook's."""
        dumped = proposed.model_dump(mode="json")
        saved = self.approvals.get(number)
        if saved is not None and saved["proposed"] == dumped:
            return NextStep.model_validate(saved["approved"])

        if self.approve is None:
            approved: NextStep | None = proposed
        else:
            approved = self.approve(number, proposed.model_copy(deep=True))
            if approved is not None and not isinstance(approved, NextStep):
                raise TypeError(f"approve must return a NextStep or None, not {type(approved).__name__}")

        if approved is not None:
            self.approvals[number] = {"proposed": dumped, "approved": approved.model_dump(mode="json")}

        return approved

    def check(self, decision: NextStep) -> str | None:
        """Why a decision cannot be carried out, or None if it can."""
        if not decision.plan:
            return "The plan was left empty. Restate the whole plan, every task with its status."

        action = decision.action
        if self.memory_mode == "pick" and action not in ("change_team",):
            if unknown := self.memory.unknown(decision.findings):
                listed = ", ".join(entry.id for entry in self.memory) or "none yet"
                return f"No finding has the id {', '.join(unknown)}. The findings: {listed}."

        if action in ("team_meeting", "individual_meeting", "write_code"):
            if not decision.agenda.strip():
                return f"{article(action)} {action} needs an agenda."
            if action == "write_code" and self.executor is None:
                return "This project has nothing to run code with, so write_code cannot be used."

            allowed = [*self.team, self.critic] if action == "team_meeting" else [self.team_lead, *self.team]
            titles = ", ".join(agent.title for agent in allowed) or "none"
            if action == "team_meeting" and not decision.participants:
                return f"A team_meeting needs at least one participant, from: {titles}."
            if action != "team_meeting" and len(decision.participants) != 1:
                return f"{article(action)} {action} needs exactly one participant, from: {titles}."
            for title in decision.participants:
                if find_agent(title, allowed) is None:
                    return f'"{title}" cannot take part in {article(action).lower()} {action}. Choose from: {titles}.'
            if len({normalize_title(title) for title in decision.participants}) != len(decision.participants):
                return "A participant was named more than once."

        elif action == "change_team":
            if not decision.add_members and not decision.remove_members:
                return "A change_team needs someone to add or remove."
            for title in decision.remove_members:
                if find_agent(title, self.team) is None:
                    members = ", ".join(agent.title for agent in self.team) or "none"
                    return f'"{title}" is not a member of the team who can be let go. Members: {members}.'
            removed = {normalize_title(title) for title in decision.remove_members}
            staying = [agent for agent in self.team if normalize_title(agent.title) not in removed]
            added = [spec.title for spec in decision.add_members]
            if any(not normalize_title(title) for title in added):
                return "A scientist to add needs a title."
            if len({normalize_title(title) for title in added}) != len(added):
                return "Two scientists to add have the same title."
            for title in added:
                if find_agent(title, [self.team_lead, self.critic, *staying]) is not None:
                    return f'"{title}" is on the project already. Give a scientist to add a title of their own.'
            if len(staying) + len(added) > self.max_team_size:
                return f"The team can have at most {self.max_team_size} members, the team lead and critic aside."

        elif not decision.answer.strip():
            return "A finish needs the project's answer."

        return None

    def take(self, number: int, decision: NextStep, round_: ProjectRound) -> tuple[Status, str] | None:
        """Takes the step decided on."""
        prefix = f"round_{number:02d}"
        query = decision.answer if decision.action == "finish" else " ".join([decision.agenda, *decision.agenda_questions])
        given = self.given(decision, query)
        round_.findings = [entry.id for entry in given]
        meeting = {
            "agenda_questions": tuple(decision.agenda_questions),
            "summaries": tuple(self.step_summaries()),
            "contexts": (self.brief(), *self.given_contexts(given)),
            "num_rounds": self.meeting_rounds,
        }
        findings = Findings if self.memory_mode != "summaries" else None

        if decision.action == "team_meeting":
            members = tuple(self.member(title, [*self.team, self.critic]) for title in decision.participants)
            round_.steps.append(f"{prefix}_meeting")
            result = self.meeting(
                "team",
                decision.agenda,
                name=f"{prefix}_meeting",
                team_lead=self.team_lead,
                team_members=members,
                output_schema=findings,
                **meeting,
            )
            who = ", ".join(agent.title for agent in members)
            self.history.append(
                Event(
                    f"Round {number}: a team meeting of {self.team_lead.title} with {who} on: {decision.agenda}",
                    result.summary,
                    shown=True,
                    work=True,
                    finding_ids=self.remember(result.output, f"{prefix}_meeting", number, round_),
                )
            )

        elif decision.action == "individual_meeting":
            member = self.member(decision.participants[0], [self.team_lead, *self.team])
            round_.steps.append(f"{prefix}_meeting")
            result = self.meeting(
                "individual",
                decision.agenda,
                name=f"{prefix}_meeting",
                team_member=member,
                critic=self.critic,
                output_schema=findings,
                **meeting,
            )
            self.history.append(
                Event(
                    f"Round {number}: {member.title} worked on: {decision.agenda}",
                    result.summary,
                    shown=True,
                    work=True,
                    finding_ids=self.remember(result.output, f"{prefix}_meeting", number, round_),
                )
            )

        elif decision.action == "write_code":
            author = self.member(decision.participants[0], [self.team_lead, *self.team])
            assert self.executor is not None
            round_.steps.append(f"{prefix}_code")
            result = self.meeting(
                "individual",
                decision.agenda,
                name=f"{prefix}_code",
                team_member=author,
                critic=self.critic,
                agenda_rules=PROJECT_CODING_RULES,
                output_schema=CodeArtifacts,
                **meeting,
            )
            assert isinstance(result.output, CodeArtifacts)
            round_.steps.append(f"{prefix}_run")
            outcome = self.project.repair(result.output, author, self.executor, name=f"{prefix}_run", **self.repair_options)
            self.emit("code", number, outcome.report(), name=f"{prefix}_run", succeeded=outcome.succeeded)
            found: tuple[str, ...] = ()
            if findings is not None:
                round_.steps.append(f"{prefix}_findings")
                stated = self.meeting(
                    "individual",
                    f"The code you wrote for this agenda was run:\n\n{decision.agenda}\n\nWhat came of running it:\n\n"
                    f"{outcome.report()}\n\nSay what the run established. If it failed, or showed nothing, say so, and "
                    "claim nothing it did not show.",
                    name=f"{prefix}_findings",
                    team_member=author,
                    contexts=(self.brief(),),
                    num_rounds=0,
                    output_schema=findings,
                )
                found = self.remember(stated.output, f"{prefix}_findings", number, round_)
            self.history.append(
                Event(
                    f"Round {number}: {author.title} wrote code for: {decision.agenda}",
                    f"{result.summary}\n\nWhat came of running it:\n\n{outcome.report()}",
                    shown=True,
                    work=True,
                    finding_ids=found,
                )
            )

        elif decision.action == "change_team":
            removed_titles = {normalize_title(title) for title in decision.remove_members}
            removed = [agent for agent in self.team if normalize_title(agent.title) in removed_titles]
            added = [spec.to_agent(model=self.team_lead.model) for spec in decision.add_members]
            self.team = [agent for agent in self.team if agent not in removed] + added
            self.record_team_change(number, added=added, removed=removed, why=decision.rationale)
            changes = [f"{describe_member(agent)}, joined" for agent in added]
            changes += [f"{agent.title} left" for agent in removed]
            self.history.append(Event(f"Round {number}: the team changed", "; ".join(changes) + ".", shown=True))

        else:
            round_.steps.append(f"{prefix}_review")
            result = self.meeting(
                "individual",
                f"The {self.team_lead.title} proposes to end the project with this answer:\n\n{decision.answer}\n\n"
                "Review it against the project's goal and the work done, as given above. The goal "
                "is met only if the answer gives everything the goal asks for, and the work supports it. Say what "
                "is missing, unsupported, or wrong, specifically enough to act on.",
                name=f"{prefix}_review",
                team_member=self.critic,
                summaries=tuple(self.step_summaries()),
                contexts=(*self.contexts(), *self.given_contexts(given)),
                num_rounds=0,
                output_schema=Review,
            )
            assert isinstance(result.output, Review)
            review = result.output
            self.objections = list(review.objections)
            if review.goal_met:
                self.answer = decision.answer
                round_.outcome = "finished"
                return "finished", f"The {self.critic.title} agreed in round {number} that the answer meets the goal."

            self.proposed_answer = decision.answer
            round_.outcome, round_.note = "objected", "; ".join(review.objections)
            objections = "\n".join(f"- {objection}" for objection in review.objections) or "- (none given)"
            self.history.append(
                Event(
                    f"Round {number}: the {self.critic.title} did not accept this answer: {decision.answer}",
                    f"The {self.critic.title}'s objections:\n{objections}",
                    shown=True,
                )
            )

        return None

    @staticmethod
    def member(title: str, pool: list[Agent]) -> Agent:
        """Who a checked decision names."""
        agent = find_agent(title, pool)
        assert agent is not None, f"{title} was checked to be among {pool}"
        return agent

    # What each step is told

    def brief(self) -> str:
        """The goal, the team, and the plan as it stands."""
        team = "\n".join(f"- {describe_member(agent)}" for agent in self.team) or "- (no one yet)"
        brief = (
            f"The project's goal:\n\n{self.project.goal}\n\n"
            f"The project is led by {describe_member(self.team_lead)}, and critiqued by "
            f"{describe_member(self.critic)}. The team:\n\n{team}"
        )
        if self.plan:
            tasks = "\n".join(f"{index}. [{task.status}] {task.task}" for index, task in enumerate(self.plan, start=1))
            brief += f"\n\nThe plan as it stands:\n\n{tasks}"

        return brief

    def summaries(self, everything: bool) -> list[str]:
        """What the steps so far found, in full: all of it for the team lead's decisions, and what
        the team is shown for the rest."""
        return [f"{event.what}\n\n{event.found}" for event in self.history if everything or event.shown]

    def decision_summaries(self) -> list[str]:
        """What the team lead is told of the steps so far when it decides the next.

        With findings kept, work is told of by the findings it made, which are listed by claim,
        except the latest, which is told of in full, as what the next step most often follows on
        from.
        """
        if self.memory_mode == "summaries":
            return self.summaries(everything=True)

        latest = max((index for index, event in enumerate(self.history) if event.work), default=None)
        told = []
        for index, event in enumerate(self.history):
            ids = f"Findings: {', '.join(event.finding_ids) or 'none'}"
            if not event.work:
                told.append(f"{event.what}\n\n{event.found}")
            elif index == latest:
                told.append(f"{event.what}\n\n{event.found}\n\n{ids}")
            else:
                told.append(f"{event.what}\n\n{ids}")

        return told

    def step_summaries(self) -> list[str]:
        """What a step is told of the steps before it: with findings kept, only what the team is
        shown of what happened to the project, since what the work found is given as findings."""
        if self.memory_mode == "summaries":
            return self.summaries(everything=False)

        return [f"{event.what}\n\n{event.found}" for event in self.history if event.shown and not event.work]

    @staticmethod
    def given_contexts(given: list[MemoryEntry]) -> tuple[str, ...]:
        """The findings a step is given, in full, as a context of their own."""
        if not given:
            return ()

        findings = "\n\n".join(entry.describe() for entry in given)
        return (f"Findings from the project's work so far, which this step builds on:\n\n{findings}",)

    def contexts(self) -> tuple[str, ...]:
        """The brief, and every finding by its claim, for the team lead's decisions and the review."""
        if self.memory_mode == "summaries" or not len(self.memory):
            return (self.brief(),)

        return (self.brief(), f"The project's findings so far, by id and claim:\n\n{self.memory.catalog()}")

    def given(self, decision: NextStep, query: str) -> list[MemoryEntry]:
        """The findings a step is given in full."""
        if decision.action == "change_team":
            return []
        if self.memory_mode == "pick":
            return self.memory.get(decision.findings)
        if self.memory_mode == "bm25":
            return self.memory.search(query, self.findings_per_step)
        return []

    def remember(self, output: Any, source: str, number: int, round_: ProjectRound) -> tuple[str, ...]:
        """Keeps what a step found, if it was asked for findings."""
        if not isinstance(output, Findings):
            return ()
        ids = tuple(entry.id for entry in self.memory.add(output.findings, source=source, round=number))
        round_.found += ids

        return ids

    # Keeping a record

    def tasks_done(self) -> int:
        return sum(task.status == "done" for task in self.plan)

    def record_team_change(self, number: int, added: list[Agent], removed: list[Agent], why: str) -> None:
        self.team_changes.append(
            {
                "round": number,
                "added": [describe_agent(agent) for agent in added],
                "removed": [agent.title for agent in removed],
                "why": why,
            }
        )
        team = [describe_agent(agent) for agent in self.team]
        self.emit("team", number or None, team=team, change=self.team_changes[-1])

    def finish_round(self, round_: ProjectRound) -> None:
        round_.team = [agent.title for agent in self.team]
        round_.tasks_done = self.tasks_done()
        plan = [task.model_dump(mode="json") for task in self.plan]
        self.emit("round", round_.number, round=asdict(round_), plan=plan)

    def describe(self, status: str, error: BaseException | None = None) -> dict[str, Any]:
        return {
            "goal": self.project.goal,
            "status": status,
            "error": {"type": type(error).__name__, "message": str(error)} if error is not None else None,
            "team_lead": describe_agent(self.team_lead),
            "critic": describe_agent(self.critic),
            "team": [describe_agent(agent) for agent in self.team],
            "team_changes": self.team_changes,
            "plan": [task.model_dump(mode="json") for task in self.plan],
            "memory": self.memory_mode,
            "rounds": [asdict(round_) for round_ in self.rounds],
            "approvals": {str(number): self.approvals[number] for number in sorted(self.approvals)},
        }

    def save_log(self, status: str = "running", error: BaseException | None = None) -> None:
        log = self.project.save_dir / RESEARCH_LOG_FILE_NAME
        write_atomically(log, json.dumps(self.describe(status, error), indent=4).encode("utf-8"))
        self.memory.save(self.project.save_dir / MEMORY_FILE_NAME)

    def finish(self, status: Status, reason: str) -> ProjectReport:
        """Ends the run: saves the research log and the report, and returns the report."""
        self.save_log(status)
        report = ProjectReport(
            goal=self.project.goal,
            status=status,
            reason=reason,
            answer=self.answer,
            proposed_answer=self.proposed_answer if self.answer is None else None,
            objections=self.objections,
            team_lead=describe_agent(self.team_lead),
            critic=describe_agent(self.critic),
            team=[describe_agent(agent) for agent in self.team],
            team_changes=self.team_changes,
            plan=[task.model_dump(mode="json") for task in self.plan],
            findings=self.memory.to_dict()["findings"],
            rounds=self.rounds,
            spent=self.project.spent,
            max_cost=self.project.max_cost,
        )
        save_dir = self.project.save_dir
        write_atomically(save_dir / REPORT_FILE_NAME, json.dumps(report.to_dict(), indent=4).encode("utf-8"))
        write_atomically(save_dir / REPORT_MARKDOWN_FILE_NAME, report.to_markdown().encode("utf-8"))
        print(f"The project ended: {status}. {reason} The report is in {save_dir / REPORT_MARKDOWN_FILE_NAME}")
        # The report is saved by now, so a failure here is not the project's
        try:
            self.emit("finished", self.rounds[-1].number if self.rounds else None, reason, report=report.to_dict())
        except Exception as event_error:
            print(f"Warning: on_event failed when told the project had ended: {event_error!r}")

        return report
