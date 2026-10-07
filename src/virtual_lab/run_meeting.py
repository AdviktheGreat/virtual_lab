"""Runs a meeting with LLM agents."""

import copy
import hashlib
import itertools
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from openai import OpenAI
from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionMessageParam, ChatCompletionToolParam
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel
from tqdm import trange, tqdm

from virtual_lab.actions import describe_tool_output, find_code_action, observation, without_code_action
from virtual_lab.agent import Agent
from virtual_lab.llm import ModelReply, ModelSource, ask, resolve_chat_models
from virtual_lab.completions import check_temperature, ran_without_temperature, send_request
from virtual_lab.events import MeetingEvent, MeetingEventKind, OnMeetingEvent
from virtual_lab.constants import (
    CONSISTENT_TEMPERATURE,
    DEFAULT_MAX_RETRIES,
    MAX_RECORDED_ARGUMENT_CHARS,
    MAX_RECORDED_RETRIEVAL_CHARS,
    MAX_TOOL_ITERATIONS,
    PARTIAL_MEETING_DIR_NAME,
    SESSION_LOG_DIR_NAME,
    SESSION_MAX_TOOL_ITERATIONS,
)
from virtual_lab.prompts import (
    code_session_prompt,
    individual_meeting_agent_prompt,
    individual_meeting_critic_prompt,
    individual_meeting_start_prompt,
    SCIENTIFIC_CRITIC,
    structured_output_prompt,
    team_meeting_start_prompt,
    team_meeting_team_lead_initial_prompt,
    team_meeting_team_lead_intermediate_prompt,
    team_meeting_team_lead_final_prompt,
    team_meeting_team_member_prompt,
)
from virtual_lab.provenance import MeetingRecord, describe_agent, save_record
from virtual_lab.records import truncate_text
from virtual_lab.resources import (
    Resources,
    available_resources,
    check_installed,
    parse_retrieval,
    resources_prompt,
    retrieval_prompt,
    select_resources,
)
from virtual_lab.session import CODE_TOOL_NAME, Session, own_resources_prompt, session_tool
from virtual_lab.structured import StructuredOutputError, request_structured_output, save_output
from virtual_lab.tools import PUBMED_TOOL, Tool, run_tool_calls, tool_instructions_prompt
from virtual_lab.utils import (
    BudgetExceededError,
    CostUnknownError,
    MeetingUsage,
    check_context_length,
    compute_token_cost,
    get_max_input_tokens,
    get_summary,
    price_per_million,
    save_meeting,
)


class TruncatedResponseError(RuntimeError):
    """Raised when a model runs out of tokens before writing any of its answer."""


@dataclass(frozen=True)
class MeetingResult:
    """Everything a meeting produced, for a caller that goes on to act on it.

    :param summary: The last thing said, which is the closing agent's summary of the meeting.
    :param output: The conclusions as an instance of the schema asked for, or None.
    :param usage: The tokens the meeting used, per model.
    :param record: How the meeting was produced, as saved alongside the transcript.
    :param discussion: The transcript, turn by turn.
    :param transcript_path: Where the transcript was saved, as JSON. The Markdown copy is beside it.
    :param record_path: Where the record was saved.
    :param output_path: Where the structured output was saved, or None if none was asked for.
    """

    summary: str
    output: BaseModel | None
    usage: MeetingUsage
    record: MeetingRecord
    discussion: tuple[dict[str, str], ...]
    transcript_path: Path
    record_path: Path
    output_path: Path | None

    @property
    def cost(self) -> float | None:
        """What the meeting cost in USD, or None if that cannot be worked out."""
        try:
            return self.usage.compute_cost()
        except CostUnknownError:
            return None


def fingerprint(text: str) -> str:
    """A hash of a piece of text, for recording which text a meeting was given."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def describe_tool_call(tool_call) -> dict[str, str]:  # type: ignore[no-untyped-def]
    """Records a tool call by name and arguments, with the arguments shortened."""
    function = getattr(tool_call, "function", None)
    name = getattr(function, "name", None) or "unknown"
    arguments = getattr(function, "arguments", None) or ""

    return {
        "name": truncate_text(str(name), MAX_RECORDED_ARGUMENT_CHARS),
        "arguments": truncate_text(str(arguments), MAX_RECORDED_ARGUMENT_CHARS),
    }


def hold_meeting(
    meeting_type: Literal["team", "individual"],
    agenda: str,
    save_dir: Path,
    save_name: str = "discussion",
    team_lead: Agent | None = None,
    team_members: tuple[Agent, ...] | None = None,
    team_member: Agent | None = None,
    critic: Agent | None = None,
    agenda_questions: tuple[str, ...] = (),
    agenda_rules: tuple[str, ...] = (),
    summaries: tuple[str, ...] = (),
    contexts: tuple[str, ...] = (),
    num_rounds: int = 0,
    temperature: float = CONSISTENT_TEMPERATURE,
    pubmed_search: bool = False,
    tools: tuple[Tool, ...] = (),
    output_schema: type[BaseModel] | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    client: OpenAI | None = None,
    chat_models: ModelSource | None = None,
    max_cost: float | None = None,
    on_usage: Callable[[MeetingUsage], None] | None = None,
    before_request: Callable[[], None] | None = None,
    max_completion_tokens: int | None = None,
    session: Session | None = None,
    code_actions: Literal["tool", "tags"] = "tool",
    max_tool_iterations: int | None = None,
    resources: Literal["retrieve", "all", "none"] | Resources = "retrieve",
    commercial_mode: bool = False,
    on_event: OnMeetingEvent | None = None,
    stream: bool = False,
) -> MeetingResult:
    """Runs a meeting with LLM agents and returns everything it produced.

    :param meeting_type: The type of meeting.
    :param agenda: The agenda for the meeting.
    :param save_dir: The directory to save the discussion.
    :param save_name: The name of the discussion file that will be saved.
    :param team_lead: The team lead for a team meeting (None for individual meeting).
    :param team_members: The team members for a team meeting (None for individual meeting).
    :param team_member: The team member for an individual meeting (None for team meeting).
    :param critic: The critic for an individual meeting (None for team meeting). If None, the
        Scientific Critic is used with the same model as team_member so that the meeting does
        not silently mix models.
    :param agenda_questions: The agenda questions to answer by the end of the meeting.
    :param agenda_rules: The rules for the meeting.
    :param summaries: The summaries of previous meetings.
    :param contexts: The contexts for the meeting.
    :param num_rounds: The number of rounds of discussion. With none, a team meeting is the team
        lead answering the agenda alone, and an individual meeting is one answer with no critic.
    :param temperature: The sampling temperature. A model that refuses it runs at its default,
        and the record says so.
    :param pubmed_search: Whether to include a PubMed search tool. Shorthand for passing
        PUBMED_TOOL in tools.
    :param tools: Additional tools the agents may call during the meeting. Make one of a
        function of your own with tool_from_function, or take an MCP server's with connect_mcp.
        To have code in a session call it instead, give it to the session.
    :param output_schema: A pydantic model for the meeting's conclusions. When given, the agent who
        closed the meeting is asked to restate them against the schema in one additional call, and
        the result is saved under save_dir/outputs/. Note that the API makes every field of the
        schema required, so a field default never applies.
    :param max_retries: The number of times to retry a failed API call, with exponential backoff.
        Only used for chat models built here.
    :param client: An OpenAI client, or one for any OpenAI-compatible server, to ask every model
        chat_models does not cover. With neither, each model is built with get_llm, which works
        out its provider from its name.
    :param chat_models: LangChain chat models to ask: one for every agent, a mapping from model
        name to chat model, or a function from model name to chat model.
    :param max_cost: The most the meeting may spend, in USD. It is checked before every request,
        so a meeting can overrun it by the cost of one request; max_completion_tokens bounds that.
        Every model in the meeting must be priced for the limit to mean anything.
    :param on_usage: Called with the meeting's usage so far after every response, so that a
        caller can keep a running total across meetings. An exception it raises stops the
        meeting the way any other failure does.
    :param before_request: Called before every request, once max_cost has been checked. An
        exception it raises stops the meeting before the request is sent, the way any other
        failure does, which lets a caller stop several meetings that share one budget.
    :param max_completion_tokens: The most tokens any one response may use, reasoning included,
        or None for the model's own limit.
    :param session: A session for the agents to run code in, shared by all of them, as in
        DockerSession or LocalSession. It is started before the first request and left running
        afterwards, so that the next meeting can use what this one computed; close it when done.
        The session's own tools, which its code can call, and its own data and software, are
        listed to the agents first and in full; its software is looked for in it first, and
        any not found is warned of, and marked as such to the agents.
        The code the meeting ran, and what came of it, is saved under save_dir/sessions/.
    :param code_actions: How the agents run code in the session: "tool" to call the run_code
        tool, or "tags" to write it between <execute> and </execute> in their replies, as
        Biomni's agent does, which works with models served without tool calling.
    :param max_tool_iterations: The most tool calls and runs of code in one agent's turn before
        it is asked to answer without them. Defaults to 20 with a session, which an analysis
        needs, and 5 without one.
    :param resources: What of Biomni's environment the agents are told they can use in the
        session: its tool functions, data lake files, software, and know-how documents. With
        "retrieve", the agent who closes the meeting is first asked which are relevant to the
        agenda, as Biomni's agent asks before each task, at the cost of one request of about
        12,000 tokens. With "all", every one is listed, which is about 40,000 tokens more in every
        request. Pass a Resources to choose them yourself, or "none" to list nothing. Only used
        with a session; tools and software are listed only where the session can import
        Biomni's tools, and then only the software the session has installed and the tools whose
        modules it can import; data is listed only where the data lake is mounted.
    :param commercial_mode: Whether to leave out data and know-how that may not be used
        commercially, as Biomni's commercial mode does. It does not check the tools' own licenses;
        see the license_info.md that ships with Biomni's package.
    :param on_event: Called with a MeetingEvent as the meeting goes: as it starts, at every turn,
        every message added to the transcript, every tool call and run of code, and every
        response's usage, and as it finishes or fails, so that it can be followed live, as in
        the web interface. An exception it raises stops the meeting the way any other failure
        does, except from the "finished" event, once the meeting is saved, and the "failed"
        event, so as not to hide the error that ended the meeting; those are warned about.
    :param stream: Whether on_event is also told of each reply as it is written, with "writing"
        events. A reply's usage is asked for with it, which OpenAI-compatible servers that do not
        report the usage of a streamed reply leave out, making its cost unknown. A server that
        refuses to be asked fails the request; give its chat model stream_usage=False.
    :raises BudgetExceededError: Before a request, if the meeting has already spent max_cost.
    :raises CostUnknownError: If max_cost is given and a model's cost, or a response's usage,
        cannot be known.
    :raises TruncatedResponseError: If a model runs out of tokens before writing anything.
    :raises Exception: If an API call fails after all retries. The completed portion of the
        discussion is saved under save_dir/partial/ before the error propagates, whatever it is.
    :return: The summary, the structured output, the usage, the record, and where each was saved.
    """
    if num_rounds < 0:
        raise ValueError(f"num_rounds must be zero or more, not {num_rounds}")

    # Infinity would be no limit at all, and is written to the record as JSON cannot hold it
    if max_cost is not None and not (math.isfinite(max_cost) and max_cost >= 0):
        raise ValueError(f"max_cost must be a finite amount, zero or more, not {max_cost}")

    check_temperature(temperature)

    if max_completion_tokens is not None and max_completion_tokens < 1:
        raise ValueError(f"max_completion_tokens must be at least 1, not {max_completion_tokens}")

    if code_actions not in ("tool", "tags"):
        raise ValueError(f'code_actions must be "tool" or "tags", not {code_actions!r}')

    if not_tools := [type(tool).__name__ for tool in tools if not isinstance(tool, Tool)]:
        raise TypeError(
            f"A meeting's tools are Tools, not {', '.join(not_tools)}: make one of a function with tool_from_function"
        )

    if code_actions == "tags" and session is None:
        raise ValueError('code_actions="tags" runs code in a session, so it needs a session')

    if stream and on_event is None:
        raise ValueError("stream tells on_event of each reply as it is written, so it needs an on_event")

    if isinstance(resources, Resources):
        if session is None:
            raise ValueError("resources lists what can be used in a session, so it needs a session")
    elif resources not in ("retrieve", "all", "none"):
        raise ValueError(f'resources must be "retrieve", "all", "none", or a Resources, not {resources!r}')

    if max_tool_iterations is None:
        max_tool_iterations = SESSION_MAX_TOOL_ITERATIONS if session is not None else MAX_TOOL_ITERATIONS
    elif max_tool_iterations < 1:
        raise ValueError(f"max_tool_iterations must be at least 1, not {max_tool_iterations}")

    # Validate meeting type
    if meeting_type == "team":
        if team_lead is None or team_members is None or len(team_members) == 0:
            raise ValueError("Team meeting requires team lead and team members")
        if team_member is not None:
            raise ValueError("Team meeting does not require individual team member")
        if team_lead in team_members:
            raise ValueError("Team lead must be separate from team members")
        if len(set(team_members)) != len(team_members):
            raise ValueError("Team members must be unique")
        if critic is not None:
            raise ValueError("Team meeting does not use a separate critic; include one in team_members")
    elif meeting_type == "individual":
        if team_member is None:
            raise ValueError("Individual meeting requires individual team member")
        if team_lead is not None or team_members is not None:
            raise ValueError("Individual meeting does not require team lead or team members")
        if critic is not None and critic.title == team_member.title:
            raise ValueError("Critic must be separate from the individual team member")
    else:
        raise ValueError(f"Invalid meeting type: {meeting_type}")

    # Start timing the meeting
    start_time = time.time()

    # Set up team
    meeting_critic: Agent | None = None

    if meeting_type == "team":
        assert team_lead is not None and team_members is not None
        team: list[Agent] = [team_lead] + list(team_members)
    else:
        assert team_member is not None
        meeting_critic = critic if critic is not None else SCIENTIFIC_CRITIC.with_model(team_member.model)
        team = [team_member, meeting_critic]

    # Set up tools, keeping pubmed_search as a shorthand for including the PubMed tool
    meeting_tools = tools

    if pubmed_search and not any(tool.name == PUBMED_TOOL.name for tool in meeting_tools):
        meeting_tools = (PUBMED_TOOL,) + meeting_tools

    if session is not None and code_actions == "tool":
        meeting_tools = meeting_tools + (session_tool(session),)

    # A limit on spending is only a limit if every model's price is known; an unpriced one
    # would be free as far as the limit could tell
    if max_cost is not None:
        for agent in team:
            try:
                compute_token_cost(agent.model, 0, 0)
            except CostUnknownError as error:
                raise CostUnknownError(
                    f"{error}, so a max_cost cannot be enforced. Add its prices to the tables in "
                    f"virtual_lab.constants, or run without a limit."
                ) from error

    duplicate_names = {tool.name for tool in meeting_tools}
    if len(duplicate_names) != len(meeting_tools):
        raise ValueError("Tool names must be unique")

    tool_definitions: list[ChatCompletionToolParam] = [tool.definition for tool in meeting_tools]

    # Built before the first request, so that a model whose provider cannot be worked out, or
    # whose package is not installed, fails the meeting before anything is spent
    llms = resolve_chat_models(
        [agent.model for agent in team], chat_models=chat_models, client=client, max_retries=max_retries
    )

    # Track the token usage reported by the API, per model
    usage = MeetingUsage()

    # Record how the meeting was produced, saved alongside the transcript
    record = MeetingRecord(
        meeting_type=meeting_type,
        save_name=save_name,
        num_rounds=num_rounds,
        temperature=temperature,
        max_retries=max_retries,
        team=[describe_agent(agent) for agent in team],
        critic=describe_agent(meeting_critic) if meeting_critic is not None else None,
        tools=[tool.name for tool in meeting_tools],
        agenda=agenda,
        agenda_questions=list(agenda_questions),
        agenda_rules=list(agenda_rules),
        summaries_sha256=[fingerprint(summary) for summary in summaries],
        contexts_sha256=[fingerprint(context) for context in contexts],
        output_schema=output_schema.__name__ if output_schema is not None else None,
        max_tool_iterations=max_tool_iterations,
        max_completion_tokens=max_completion_tokens,
        max_cost=max_cost,
        chat_models={model: describe_chat_model(llm) for model, llm in llms.items()},
    )

    # The round events are told of in, from 1, and None outside the rounds
    event_round: int | None = None
    request_numbers = itertools.count(1)

    # The kind is positional only, so that a message's own kind can be told of in data
    def emit(kind: MeetingEventKind, /, speaker: str | None = None, text: str = "", **data: Any) -> None:
        if on_event is not None:
            event = MeetingEvent(kind=kind, meeting=save_name, round=event_round, speaker=speaker, text=text, data=data)
            on_event(event)

    def check_budget() -> None:
        """Stops the meeting before a request it has no money left for."""
        if max_cost is not None and (spent := usage.compute_cost()) >= max_cost:
            raise BudgetExceededError(spent=spent, limit=max_cost)

        if before_request is not None:
            before_request()

    def count_usage(model: str, reported, turn_usage: MeetingUsage) -> None:  # type: ignore[no-untyped-def]
        """Adds a response's usage to the meeting and the turn, and tells the caller."""
        usage.add(model=model, usage=reported)
        turn_usage.add(model=model, usage=reported)

        if on_usage is not None:
            on_usage(usage)

        emit("usage", **usage.to_dict())

    def ask_agent(
        agent: Agent,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam] | None,
        speaking: bool = True,
    ) -> ModelReply:
        """Asks an agent for its next message, without a temperature if its model refuses one.

        :param speaking: Whether the reply is the agent's say in the meeting, which is streamed
            to on_event with stream, rather than a request made for the meeting's sake.
        """
        on_text = None
        if stream and speaking:
            request = next(request_numbers)

            def on_text(text: str) -> None:
                emit("writing", speaker=agent.title, text=text, request=request, model=agent.model)

        return send_request(
            lambda sent_temperature: ask(
                llms[agent.model],
                messages,
                temperature=sent_temperature,
                tools=tools or None,
                max_tokens=max_completion_tokens,
                reader=agent.name,
                on_text=on_text,
            ),
            model=agent.model,
            temperature=temperature,
        )

    def add_turn(speaker: str, message: str, kind: str, **recorded: Any) -> None:
        """Adds a message to the transcript and the record, and tells on_event of it."""
        discussion.append({"agent": speaker, "message": message})
        record.record_turn(speaker=speaker, kind=kind, **recorded)
        emit("message", speaker=speaker, text=message, kind=kind, index=len(discussion) - 1)

    def tell_cells() -> None:
        """Tells on_event of the code the session has run since it was last told."""
        nonlocal cells_told
        if session is None:
            return
        for cell in session.history[cells_told:]:
            emit("cell", **cell.to_dict(), plot_paths=[str(session.directory / plot) for plot in cell.plots])
        cells_told = len(session.history)

    # Started before the first request, so that a session that cannot start, such as one whose
    # image was never built, fails the meeting before anything is spent. The session may have
    # run code for an earlier meeting; only what this one runs is its own.
    first_cell = 0
    catalog: Resources | None = None
    software_not_found: list[str] | None = None
    if session is not None:
        session.start()
        first_cell = len(session.history)
        if session.software:
            checked = check_installed(session, list(session.software), [], what="the software added to it")
            if checked is not None:
                software_not_found = [name for name in session.software if name in checked[0]]
                if software_not_found:
                    print(
                        f"Warning: the session does not have {', '.join(software_not_found)}: not as a "
                        "Python distribution or module, an R package, or a command. The agents are "
                        "told of it anyway, as not found."
                    )
        if isinstance(resources, Resources):
            catalog = resources
            if catalog.data_lake and catalog.data_lake_path is None:
                catalog = replace(catalog, data_lake_path=session.data_lake_path())
        elif resources != "none":
            catalog = available_resources(session, commercial_mode=commercial_mode)

    if catalog is not None:
        record.resources = {
            "mode": resources if isinstance(resources, str) else "given",
            "commercial_mode": commercial_mode,
            "available": catalog.counts(),
            "not_installed": catalog.not_installed,
            "selected": catalog.names(),
            "retrieval": None,
        }

    def choose_resources() -> Resources | None:
        """Asks the agent who closes the meeting which resources the agenda needs, as Biomni does."""
        if catalog is None or resources != "retrieve" or catalog.is_empty():
            return catalog

        chooser = team[0]
        query = agenda
        if agenda_questions:
            query += "\n\n" + "\n".join(f"{number}. {question}" for number, question in enumerate(agenda_questions, 1))

        retrieval_messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": retrieval_prompt(query, catalog)}
        ]
        check_context_length(messages=retrieval_messages, model=chooser.model)
        check_budget()
        retrieval_usage = MeetingUsage()
        reply = ask_agent(agent=chooser, messages=retrieval_messages, tools=None, speaking=False)
        count_usage(chooser.model, reply.usage, retrieval_usage)

        chosen = parse_retrieval(reply.content)
        if chosen is None:
            # An answer with no list at all is not a choice of nothing, and listing nothing
            # would leave the agents without the tools the meeting was given a session for
            print(
                f"Warning: {chooser.title} did not say which resources the agenda needs, so all of "
                f"them are listed."
            )
            selected = catalog
        else:
            selected = select_resources(catalog, chosen)

        assert record.resources is not None
        record.resources["selected"] = selected.names()
        record.resources["retrieval"] = {
            "model": chooser.model,
            "name": chooser.name,
            "understood": chosen is not None,
            "reply": truncate_text(reply.content, MAX_RECORDED_RETRIEVAL_CHARS),
            "input_tokens": retrieval_usage.input_tokens,
            "cached_input_tokens": retrieval_usage.cached_input_tokens,
            "output_tokens": retrieval_usage.output_tokens,
            "reasoning_tokens": retrieval_usage.reasoning_tokens,
            "system_fingerprint": reply.system_fingerprint,
            "finish_reason": reply.finish_reason,
        }
        listed = ", ".join(f"{count} {kind.replace('_', ' ')}" for kind, count in selected.counts().items())
        print(f"Resources listed for the session: {listed}")

        return selected

    def save_session(directory: Path) -> None:
        """Saves the code the meeting ran, and notes in the record where it went.

        A log that cannot be saved is warned about rather than raised, so that it neither costs
        the meeting its transcript nor hides the error that ended a failed one.
        """
        if session is None:
            return

        try:
            log = save_session_log(directory, save_name, session, first_cell)
            record.session = {
                **session.describe(),
                "code_actions": code_actions,
                "cells_run": len(session.history) - first_cell,
                "log": log.relative_to(directory).as_posix(),
                **({"software_not_found": software_not_found} if session.software else {}),
            }
        except Exception as error:
            print(f"Warning: the log of the code this meeting ran could not be saved: {error}")

    # The last thing an agent said, kept apart from the transcript because a structured output
    # follows it there
    summary = ""

    # Warn only on the first request that approaches the model's input limit
    warned_about_context = False

    # Initialize discussion (list of agent/message dicts for output)
    discussion: list[dict[str, str]] = []

    # Set only when output_schema is given, and returned in place of the summary
    structured_output: BaseModel | None = None

    # Initialize messages for API calls
    messages: list[ChatCompletionMessageParam] = []

    # The cells of the session's history on_event has been told of
    cells_told = first_cell

    emit(
        "started",
        meeting_type=meeting_type,
        agenda=agenda,
        agenda_questions=list(agenda_questions),
        num_rounds=num_rounds,
        team=[describe_agent(agent) for agent in team],
        tools=[tool.name for tool in meeting_tools],
        max_cost=max_cost,
        session=session.describe() if session is not None else None,
    )

    try:
        session_prompt = ""
        if session is not None:
            selected = choose_resources()
            if record.resources is not None:
                emit("resources", **copy.deepcopy(record.resources))
            session_prompt = code_session_prompt(
                code_actions=code_actions,
                working_directory=session.where_code_runs(),
                network=session.can_reach_network(),
                data_lake=session.data_lake_path(),
            )
            if own := own_resources_prompt(session, software_not_found or ()):
                session_prompt = f"{session_prompt}\n\n{own}"
            retrieved = record.resources is not None and bool((record.resources["retrieval"] or {}).get("understood"))
            if selected is not None and (listed := resources_prompt(selected, code_actions, retrieved=retrieved)):
                session_prompt = f"{session_prompt}\n\n{listed}"
        # The tools the agents call and those their code does are told of together, since tools
        # that share instructions, such as one MCP server's, can be given to both
        session_tools = session.tools if session is not None else ()
        if instructions := tool_instructions_prompt((*meeting_tools, *session_tools)):
            session_prompt = f"{session_prompt}\n\n{instructions}" if session_prompt else instructions

        # Initial prompt for team meeting
        if meeting_type == "team":
            assert team_lead is not None and team_members is not None
            initial_content = team_meeting_start_prompt(
                team_lead=team_lead,
                team_members=team_members,
                agenda=agenda,
                agenda_questions=agenda_questions,
                agenda_rules=agenda_rules,
                summaries=summaries,
                contexts=contexts,
                num_rounds=num_rounds,
            )
            if session_prompt:
                initial_content = f"{initial_content}\n\n{session_prompt}"
            messages.append({"role": "user", "content": initial_content})
            add_turn("User", initial_content, "prompt")

        # Loop through rounds
        for round_index in trange(num_rounds + 1, desc="Rounds (+ Final Round)"):
            round_num = round_index + 1
            event_round = round_num

            # Loop through team and elicit responses
            for agent in tqdm(team, desc="Team"):
                # Prompt based on agent and round number
                if meeting_type == "team":
                    assert team_lead is not None
                    # Team meeting prompts
                    # Compared by identity because two distinct agents can share a title
                    if agent is team_lead:
                        # The final round is checked first. With no rounds of discussion the
                        # first round is also the last, and asking for opening thoughts there
                        # ended the meeting on questions for a team that never got to speak.
                        if round_index == num_rounds:
                            prompt = team_meeting_team_lead_final_prompt(
                                team_lead=team_lead,
                                agenda=agenda,
                                agenda_questions=agenda_questions,
                                agenda_rules=agenda_rules,
                            )
                        elif round_index == 0:
                            prompt = team_meeting_team_lead_initial_prompt(team_lead=team_lead)
                        else:
                            prompt = team_meeting_team_lead_intermediate_prompt(
                                team_lead=team_lead,
                                round_num=round_num - 1,
                                num_rounds=num_rounds,
                            )
                    else:
                        prompt = team_meeting_team_member_prompt(
                            team_member=agent, round_num=round_num, num_rounds=num_rounds
                        )
                else:
                    assert team_member is not None and meeting_critic is not None
                    # Individual meeting prompts
                    if agent is meeting_critic:
                        prompt = individual_meeting_critic_prompt(critic=meeting_critic, agent=team_member)
                    else:
                        if round_index == 0:
                            prompt = individual_meeting_start_prompt(
                                team_member=team_member,
                                agenda=agenda,
                                agenda_questions=agenda_questions,
                                agenda_rules=agenda_rules,
                                summaries=summaries,
                                contexts=contexts,
                            )
                            if session_prompt:
                                prompt = f"{prompt}\n\n{session_prompt}"
                        else:
                            prompt = individual_meeting_agent_prompt(critic=meeting_critic, agent=team_member)

                # Add prompt as user message
                messages.append({"role": "user", "content": prompt})
                add_turn("User", prompt, "prompt")
                emit("turn", speaker=agent.title, name=agent.name, model=agent.model)

                # A turn can span several API calls when the agent uses tools, so its usage is
                # accumulated separately from the meeting total
                turn_usage = MeetingUsage()
                turn_tool_calls: list[dict[str, str]] = []
                turn_fingerprint: str | None = None
                turn_first_cell = len(session.history) if session is not None else 0

                # Build messages for this agent with their system prompt
                agent_messages: list[ChatCompletionMessageParam] = [agent.message] + messages

                # Fail before paying for a request that cannot fit
                estimated_tokens, near_limit = check_context_length(
                    messages=agent_messages, model=agent.model
                )
                if near_limit and not warned_about_context:
                    warned_about_context = True
                    print(
                        f"Warning: this meeting is using about {estimated_tokens:,} of the "
                        f"{get_max_input_tokens(agent.model):,} input tokens available to "
                        f'"{agent.model}" and may not fit for many more rounds.'
                    )

                # Call the chat completions API, letting the agent use tools, or run code, repeatedly
                # until it has what it needs. Tool definitions are offered again after each result
                # so that one search can inform the next.
                for tool_iteration in range(max_tool_iterations + 1):
                    # Withhold the tools on the final attempt to force a text answer
                    is_final_attempt = tool_iteration == max_tool_iterations

                    check_budget()
                    reply = ask_agent(
                        agent=agent,
                        messages=agent_messages,
                        tools=tool_definitions if not is_final_attempt else None,
                    )
                    count_usage(agent.model, reply.usage, turn_usage)
                    turn_fingerprint = reply.system_fingerprint or turn_fingerprint
                    finish_reason = reply.finish_reason

                    action = (
                        find_code_action(reply.content)
                        if code_actions == "tags" and not reply.tool_calls
                        else None
                    )

                    # Stop once the agent has answered, and never run tools on the forced attempt
                    if (not reply.tool_calls and action is None) or is_final_attempt:
                        break

                    if tool_iteration == max_tool_iterations - 1:
                        print(
                            f"Warning: {agent.title} reached the limit of {max_tool_iterations} "
                            f"rounds of tool calls and will now be asked to answer without tools."
                        )

                    if action is not None:
                        assert session is not None
                        emit("code", speaker=agent.title, text=action.said, code=action.code, language=action.language)
                        try:
                            report = observation(
                                session.run(action.code, language=action.language),
                                runs_left=max_tool_iterations - tool_iteration - 1,
                            )
                        except Exception as error:
                            report = f"<observation>\nThe code could not be run: {error}\n</observation>"
                        tell_cells()

                        # Kept in the meeting as the agent said it, so that everyone after sees
                        # the code and what it printed
                        messages.append({"role": "assistant", "name": agent.name, "content": action.said})
                        add_turn(agent.title, action.said, "code_action", name=agent.name, model=agent.model)

                        messages.append({"role": "user", "content": report})
                        add_turn("Session", report, "code_output")
                    else:
                        turn_tool_calls.extend(describe_tool_call(tool_call) for tool_call in reply.tool_calls)

                        emit(
                            "tool_calls",
                            speaker=agent.title,
                            text=reply.content,
                            calls=[
                                {"id": call.id, "name": call.function.name, "arguments": call.function.arguments}
                                for call in reply.tool_calls
                            ],
                        )

                        # Run the tools and get outputs
                        tool_outputs, tool_messages = run_tool_calls(
                            tool_calls=list(reply.tool_calls), tools=meeting_tools
                        )
                        tell_cells()

                        # Add the assistant's message with tool_calls to the messages
                        assistant_tool_message: ChatCompletionAssistantMessageParam = {
                            "role": "assistant",
                            "name": agent.name,
                            "content": reply.content or None,
                            "tool_calls": [tc.model_dump() for tc in reply.tool_calls],  # type: ignore[misc]
                        }
                        messages.append(assistant_tool_message)

                        # Add tool response messages
                        for tool_msg in tool_messages:
                            messages.append(tool_msg)

                        # Add tool outputs to discussion for visibility, with the code that
                        # produced any output of code, which the model is not sent again
                        tool_output_content = "\n\n".join(
                            describe_tool_output(tool_call, output, CODE_TOOL_NAME)
                            for tool_call, output in zip(reply.tool_calls, tool_outputs)
                        )
                        add_turn("Tool", tool_output_content, "tool_output")

                        # Code written beside a tool call is not run, and the agent must know
                        # that, or it will take the call's output for the code's
                        if code_actions == "tags" and find_code_action(reply.content or "") is not None:
                            skipped = (
                                "<observation>\nThe code in that reply was not run, because the "
                                "reply also called a tool. Write it again, on its own, to run it."
                                "\n</observation>"
                            )
                            messages.append({"role": "user", "content": skipped})
                            add_turn("Session", skipped, "code_output")

                    # Send the results back on the next iteration
                    agent_messages = [agent.message] + messages

                    # Tool output can be large, so re-check before sending it back
                    check_context_length(messages=agent_messages, model=agent.model)

                # Extract the response content
                response_content = reply.content

                # Only the forced final attempt can end a turn on code, which it was too late to run
                if code_actions == "tags":
                    response_content = without_code_action(response_content)

                # A reasoning model can spend its whole allowance thinking and write nothing, and
                # an empty turn read as the agent's answer would be summarised as agreement
                if finish_reason == "length" and not response_content.strip():
                    raise TruncatedResponseError(
                        f"{agent.title} ran out of tokens before writing any of its answer. "
                        f"Allow more with max_completion_tokens, or use fewer rounds."
                    )

                if finish_reason == "length":
                    print(f"Warning: {agent.title} ran out of tokens and its answer is cut short.")

                summary = response_content

                # Add response to messages and discussion. The author name is what lets the other
                # agents tell whose turn they are reading; without it every prior turn arrives as
                # the reader's own words, which pushes the whole meeting towards agreement.
                messages.append({"role": "assistant", "name": agent.name, "content": response_content})
                add_turn(
                    agent.title,
                    response_content,
                    "response",
                    name=agent.name,
                    model=agent.model,
                    input_tokens=turn_usage.input_tokens,
                    cached_input_tokens=turn_usage.cached_input_tokens,
                    output_tokens=turn_usage.output_tokens,
                    reasoning_tokens=turn_usage.reasoning_tokens,
                    num_api_calls=turn_usage.num_calls,
                    tool_calls=turn_tool_calls,
                    code_runs=describe_code_runs(session, turn_first_cell, first_cell),
                    system_fingerprint=turn_fingerprint,
                    finish_reason=finish_reason,
                )

                # If final round, only team lead or team member responds
                if round_index == num_rounds:
                    break

        event_round = None

        # Ask whoever closed the meeting to restate its conclusions against the schema. This is a
        # separate pass rather than a constraint on the final turn so that the structured answer
        # is drawn from the complete meeting, including that final summary.
        if output_schema is not None:
            closing_agent = team[0]
            extraction_prompt = structured_output_prompt(agent=closing_agent)
            messages.append({"role": "user", "content": extraction_prompt})
            add_turn("User", extraction_prompt, "prompt")

            agent_messages = [closing_agent.message] + messages
            check_context_length(messages=agent_messages, model=closing_agent.model)

            check_budget()
            extraction_usage = MeetingUsage()

            try:
                structured_output, structured_reply = request_structured_output(
                    llm=llms[closing_agent.model],
                    model=closing_agent.model,
                    messages=agent_messages,
                    schema=output_schema,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                    reader=closing_agent.name,
                )
            except StructuredOutputError as error:
                count_usage(closing_agent.model, error.usage, extraction_usage)
                raise

            count_usage(closing_agent.model, structured_reply.usage, extraction_usage)

            output_json = json.dumps(structured_output.model_dump(mode="json"), indent=4)
            add_turn(
                closing_agent.title,
                output_json,
                "structured_output",
                name=closing_agent.name,
                model=closing_agent.model,
                input_tokens=extraction_usage.input_tokens,
                cached_input_tokens=extraction_usage.cached_input_tokens,
                output_tokens=extraction_usage.output_tokens,
                reasoning_tokens=extraction_usage.reasoning_tokens,
                num_api_calls=extraction_usage.num_calls,
                system_fingerprint=structured_reply.system_fingerprint,
                finish_reason=structured_reply.finish_reason,
            )
    # BaseException, so that a meeting interrupted from the keyboard or cancelled by whatever
    # was running it still leaves what it had done, and what it had spent, on disk
    except BaseException as error:
        close_record(record, usage, team)
        # A meeting can be many minutes and many dollars of work, so preserve whatever
        # completed. It goes in a subdirectory rather than alongside the finished meetings
        # because callers glob patterns like "discussion_*.json" and feed the last turn of
        # every match to load_summaries, which would treat a truncated meeting as a summary.
        record.finish(usage=usage, elapsed_time=time.time() - start_time, error=error)

        partial_path = None
        if discussion:
            partial_dir = save_dir / PARTIAL_MEETING_DIR_NAME
            save_meeting(save_dir=partial_dir, save_name=save_name, discussion=discussion)
            save_session(partial_dir)
            save_record(save_dir=partial_dir, save_name=save_name, record=record)
            partial_path = partial_dir / f"{save_name}.json"
            print(f"Meeting failed. Partial discussion saved to {partial_path}")
        print("Usage before the failure:")
        usage.print_summary(elapsed_time=time.time() - start_time)
        event_round = None
        try:
            emit(
                "failed",
                text=str(error) or type(error).__name__,
                type=type(error).__name__,
                partial_path=str(partial_path) if partial_path is not None else None,
                usage=usage.to_dict(),
            )
        except Exception as event_error:
            print(f"Warning: on_event failed when told the meeting had failed: {event_error!r}")
        raise

    close_record(record, usage, team)
    record.finish(usage=usage, elapsed_time=time.time() - start_time, error=None)

    # Print the usage reported by the API, priced per model
    usage.print_summary(elapsed_time=time.time() - start_time)

    # Save the discussion as JSON and Markdown, plus the record of what produced it
    save_meeting(
        save_dir=save_dir,
        save_name=save_name,
        discussion=discussion,
    )
    save_session(save_dir)
    record_path = save_record(save_dir=save_dir, save_name=save_name, record=record)
    output_path = (
        save_output(save_dir=save_dir, save_name=save_name, output=structured_output)
        if structured_output is not None
        else None
    )
    transcript_path = save_dir / f"{save_name}.json"

    # The meeting is saved and paid for by now, so a failure here is not the meeting's
    try:
        emit(
            "finished",
            text=summary,
            transcript_path=str(transcript_path),
            record_path=str(record_path),
            output_path=str(output_path) if output_path is not None else None,
            output=structured_output.model_dump(mode="json") if structured_output is not None else None,
            usage=usage.to_dict(),
            elapsed_time=record.elapsed_seconds,
        )
    except Exception as event_error:
        print(f"Warning: on_event failed when told the meeting had finished: {event_error!r}")

    return MeetingResult(
        summary=summary,
        output=structured_output,
        usage=usage,
        record=record,
        discussion=tuple(discussion),
        transcript_path=transcript_path,
        record_path=record_path,
        output_path=output_path,
    )


def describe_code_runs(session: Session | None, turn_first_cell: int, first_cell: int) -> list[dict]:
    """Records the code a turn ran, by its position in the meeting's session log."""
    if session is None:
        return []

    turn_cells = session.history[turn_first_cell:]

    return [
        {
            "cell": cell,
            "language": result.language,
            "status": result.status,
            "duration": round(result.duration, 3),
            "plots": list(result.plots),
        }
        for cell, result in enumerate(turn_cells, start=turn_first_cell - first_cell)
    ]


def save_session_log(save_dir: Path, save_name: str, session: Session, first_cell: int) -> Path:
    """Saves the code a meeting ran in its session, cell by cell, with everything it printed.

    :param save_dir: The directory the meeting is saved in.
    :param save_name: The meeting's name.
    :param session: The session.
    :param first_cell: How many cells the session had run before the meeting.
    :return: Where the log was saved.
    """
    path = save_dir / SESSION_LOG_DIR_NAME / f"{save_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    log = {
        "session": session.describe(),
        "cells": [result.to_dict() for result in session.history[first_cell:]],
    }
    path.write_text(json.dumps(log, indent=4), encoding="utf-8")

    return path


def describe_chat_model(llm: BaseChatModel) -> dict[str, str | None]:
    """Says which provider answered for a model, for the record of a meeting."""
    base_url = next(
        (
            getattr(llm, field)
            for field in ("openai_api_base", "anthropic_api_url", "base_url", "azure_endpoint", "endpoint_url")
            if getattr(llm, field, None)
        ),
        None,
    )

    if base_url is not None:
        # A server's address can carry a password, which has no place in a record
        parts = urlsplit(str(base_url))
        base_url = urlunsplit(parts._replace(netloc=parts.netloc.rpartition("@")[2]))

    return {"class": f"{type(llm).__module__}.{type(llm).__qualname__}", "base_url": base_url}


def close_record(record: MeetingRecord, usage: MeetingUsage, team: list[Agent]) -> None:
    """Adds to a record what is only known once the meeting has stopped.

    :param record: The meeting's record.
    :param usage: The usage accumulated over the meeting.
    :param team: Everyone who spoke.
    """
    models = sorted({agent.model for agent in team} | set(usage.per_model))
    record.models_at_default_temperature = ran_without_temperature(models)
    record.prices = {model: price_per_million(model) for model in models}


def run_meeting(
    meeting_type: Literal["team", "individual"],
    agenda: str,
    save_dir: Path,
    save_name: str = "discussion",
    team_lead: Agent | None = None,
    team_members: tuple[Agent, ...] | None = None,
    team_member: Agent | None = None,
    critic: Agent | None = None,
    agenda_questions: tuple[str, ...] = (),
    agenda_rules: tuple[str, ...] = (),
    summaries: tuple[str, ...] = (),
    contexts: tuple[str, ...] = (),
    num_rounds: int = 0,
    temperature: float = CONSISTENT_TEMPERATURE,
    pubmed_search: bool = False,
    tools: tuple[Tool, ...] = (),
    output_schema: type[BaseModel] | None = None,
    return_summary: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> str | BaseModel | None:
    """Runs a meeting with LLM agents.

    The original interface, kept for the notebooks written against it. It takes the arguments
    hold_meeting does, which documents them, and returns less: code that goes on to act on a
    meeting, or to limit what it spends, should call hold_meeting instead.

    :param return_summary: Whether to return the summary of the meeting.
    :return: The structured output if output_schema is given, else the summary of the meeting
        (i.e., the last message) if return_summary is True, else None.
    """
    # A structured output replaces the summary as the return value, so asking for both is a
    # mistake about which one the caller will get
    if output_schema is not None and return_summary:
        raise ValueError("Use either output_schema or return_summary, not both")

    result = hold_meeting(
        meeting_type=meeting_type,
        agenda=agenda,
        save_dir=save_dir,
        save_name=save_name,
        team_lead=team_lead,
        team_members=team_members,
        team_member=team_member,
        critic=critic,
        agenda_questions=agenda_questions,
        agenda_rules=agenda_rules,
        summaries=summaries,
        contexts=contexts,
        num_rounds=num_rounds,
        temperature=temperature,
        pubmed_search=pubmed_search,
        tools=tools,
        output_schema=output_schema,
        max_retries=max_retries,
    )

    if result.output is not None:
        return result.output

    if return_summary:
        return get_summary(list(result.discussion))

    return None
