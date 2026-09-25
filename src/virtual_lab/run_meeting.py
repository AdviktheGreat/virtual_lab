"""Runs a meeting with LLM agents."""

import json
import time
from pathlib import Path
from typing import Literal

from openai import OpenAI, NOT_GIVEN
from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionMessageParam, ChatCompletionToolParam
from pydantic import BaseModel
from tqdm import trange, tqdm

from virtual_lab.agent import Agent
from virtual_lab.constants import (
    CONSISTENT_TEMPERATURE,
    DEFAULT_MAX_RETRIES,
    MAX_TOOL_ITERATIONS,
    PARTIAL_MEETING_DIR_NAME,
)
from virtual_lab.prompts import (
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
from virtual_lab.structured import request_structured_output, save_output
from virtual_lab.tools import PUBMED_TOOL, Tool, run_tool_calls
from virtual_lab.utils import (
    MeetingUsage,
    check_context_length,
    get_max_input_tokens,
    get_summary,
    save_meeting,
)


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
    """Runs a meeting with a LLM agents.

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
    :param num_rounds: The number of rounds of discussion.
    :param temperature: The sampling temperature.
    :param pubmed_search: Whether to include a PubMed search tool. Shorthand for passing
        PUBMED_TOOL in tools.
    :param tools: Additional tools the agents may call during the meeting.
    :param output_schema: A pydantic model for the meeting's conclusions. When given, the agent who
        closed the meeting is asked to restate them against the schema in one additional call, the
        result is saved under save_dir/outputs/, and the validated instance is returned.
    :param return_summary: Whether to return the summary of the meeting.
    :param max_retries: The number of times to retry a failed API call, with exponential backoff.
    :raises Exception: If an API call fails after all retries. The completed portion of the
        discussion is saved under save_dir/partial/ before the error propagates.
    :return: The structured output if output_schema is given, else the summary of the meeting
        (i.e., the last message) if return_summary is True, else None.
    """
    # A structured output replaces the summary as the return value, so asking for both is a
    # mistake about which one the caller will get
    if output_schema is not None and return_summary:
        raise ValueError("Use either output_schema or return_summary, not both")

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

    # Set up client
    client = OpenAI(max_retries=max_retries)

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

    duplicate_names = {tool.name for tool in meeting_tools}
    if len(duplicate_names) != len(meeting_tools):
        raise ValueError("Tool names must be unique")

    tool_definitions: list[ChatCompletionToolParam] = [tool.definition for tool in meeting_tools]

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
    )

    # Warn only on the first request that approaches the model's input limit
    warned_about_context = False

    # Initialize discussion (list of agent/message dicts for output)
    discussion: list[dict[str, str]] = []

    # Set only when output_schema is given, and returned in place of the summary
    structured_output: BaseModel | None = None

    # Initialize messages for API calls
    messages: list[ChatCompletionMessageParam] = []

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
        messages.append({"role": "user", "content": initial_content})
        discussion.append({"agent": "User", "message": initial_content})
        record.record_turn(speaker="User", kind="prompt")

    # Loop through rounds
    try:
        for round_index in trange(num_rounds + 1, desc="Rounds (+ Final Round)"):
            round_num = round_index + 1

            # Loop through team and elicit responses
            for agent in tqdm(team, desc="Team"):
                # Prompt based on agent and round number
                if meeting_type == "team":
                    assert team_lead is not None
                    # Team meeting prompts
                    # Compared by identity because two distinct agents can share a title
                    if agent is team_lead:
                        if round_index == 0:
                            prompt = team_meeting_team_lead_initial_prompt(team_lead=team_lead)
                        elif round_index == num_rounds:
                            prompt = team_meeting_team_lead_final_prompt(
                                team_lead=team_lead,
                                agenda=agenda,
                                agenda_questions=agenda_questions,
                                agenda_rules=agenda_rules,
                            )
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
                        else:
                            prompt = individual_meeting_agent_prompt(critic=meeting_critic, agent=team_member)

                # Add prompt as user message
                messages.append({"role": "user", "content": prompt})
                discussion.append({"agent": "User", "message": prompt})
                record.record_turn(speaker="User", kind="prompt")

                # A turn can span several API calls when the agent uses tools, so its usage is
                # accumulated separately from the meeting total
                turn_usage = MeetingUsage()
                turn_tool_calls: list[str] = []
                turn_fingerprint: str | None = None

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

                # Call the chat completions API, letting the agent use tools repeatedly until it
                # has what it needs. Tool definitions are offered again after each result so
                # that one search can inform the next.
                for tool_iteration in range(MAX_TOOL_ITERATIONS + 1):
                    # Withhold the tools on the final attempt to force a text answer
                    is_final_attempt = tool_iteration == MAX_TOOL_ITERATIONS

                    response = client.chat.completions.create(
                        model=agent.model,
                        messages=agent_messages,
                        temperature=temperature,
                        tools=tool_definitions if tool_definitions and not is_final_attempt else NOT_GIVEN,
                    )
                    usage.add(model=agent.model, usage=response.usage)
                    turn_usage.add(model=agent.model, usage=response.usage)
                    turn_fingerprint = response.system_fingerprint or turn_fingerprint
                    response_message = response.choices[0].message

                    # Stop once the agent has answered, and never run tools on the forced attempt
                    if not response_message.tool_calls or is_final_attempt:
                        break

                    turn_tool_calls.extend(
                        tool_call.function.name for tool_call in response_message.tool_calls
                    )

                    if tool_iteration == MAX_TOOL_ITERATIONS - 1:
                        print(
                            f"Warning: {agent.title} reached the limit of {MAX_TOOL_ITERATIONS} "
                            f"rounds of tool calls and will now be asked to answer without tools."
                        )

                    # Run the tools and get outputs
                    tool_outputs, tool_messages = run_tool_calls(
                        tool_calls=response_message.tool_calls, tools=meeting_tools
                    )

                    # Add the assistant's message with tool_calls to the messages
                    assistant_tool_message: ChatCompletionAssistantMessageParam = {
                        "role": "assistant",
                        "name": agent.name,
                        "content": response_message.content,
                        "tool_calls": [tc.model_dump() for tc in response_message.tool_calls],  # type: ignore[misc]
                    }
                    messages.append(assistant_tool_message)

                    # Add tool response messages
                    for tool_msg in tool_messages:
                        messages.append(tool_msg)

                    # Add tool outputs to discussion for visibility
                    tool_output_content = "\n\n".join(tool_outputs)
                    discussion.append({"agent": "Tool", "message": tool_output_content})
                    record.record_turn(speaker="Tool", kind="tool_output")

                    # Send the tool results back on the next iteration
                    agent_messages = [agent.message] + messages

                    # Tool output can be large, so re-check before sending it back
                    check_context_length(messages=agent_messages, model=agent.model)

                # Extract the response content
                response_content = response_message.content or ""

                # Add response to messages and discussion. The author name is what lets the other
                # agents tell whose turn they are reading; without it every prior turn arrives as
                # the reader's own words, which pushes the whole meeting towards agreement.
                messages.append({"role": "assistant", "name": agent.name, "content": response_content})
                discussion.append({"agent": agent.title, "message": response_content})
                record.record_turn(
                    speaker=agent.title,
                    kind="response",
                    name=agent.name,
                    model=agent.model,
                    input_tokens=turn_usage.input_tokens,
                    cached_input_tokens=turn_usage.cached_input_tokens,
                    output_tokens=turn_usage.output_tokens,
                    reasoning_tokens=turn_usage.reasoning_tokens,
                    num_api_calls=turn_usage.num_calls,
                    tool_calls=turn_tool_calls,
                    system_fingerprint=turn_fingerprint,
                )

                # If final round, only team lead or team member responds
                if round_index == num_rounds:
                    break

        # Ask whoever closed the meeting to restate its conclusions against the schema. This is a
        # separate pass rather than a constraint on the final turn so that the structured answer
        # is drawn from the complete meeting, including that final summary.
        if output_schema is not None:
            closing_agent = team[0]
            extraction_prompt = structured_output_prompt(agent=closing_agent)
            messages.append({"role": "user", "content": extraction_prompt})
            discussion.append({"agent": "User", "message": extraction_prompt})
            record.record_turn(speaker="User", kind="prompt")

            agent_messages = [closing_agent.message] + messages
            check_context_length(messages=agent_messages, model=closing_agent.model)

            structured_output, parsed_response = request_structured_output(
                client=client,
                model=closing_agent.model,
                messages=agent_messages,
                schema=output_schema,
                temperature=temperature,
            )
            usage.add(model=closing_agent.model, usage=parsed_response.usage)

            extraction_usage = MeetingUsage()
            extraction_usage.add(model=closing_agent.model, usage=parsed_response.usage)

            output_json = json.dumps(structured_output.model_dump(mode="json"), indent=4)
            discussion.append({"agent": closing_agent.title, "message": output_json})
            record.record_turn(
                speaker=closing_agent.title,
                kind="structured_output",
                name=closing_agent.name,
                model=closing_agent.model,
                input_tokens=extraction_usage.input_tokens,
                cached_input_tokens=extraction_usage.cached_input_tokens,
                output_tokens=extraction_usage.output_tokens,
                reasoning_tokens=extraction_usage.reasoning_tokens,
                num_api_calls=extraction_usage.num_calls,
                system_fingerprint=parsed_response.system_fingerprint,
            )
    except Exception as error:
        # A meeting can be many minutes and many dollars of work, so preserve whatever
        # completed. It goes in a subdirectory rather than alongside the finished meetings
        # because callers glob patterns like "discussion_*.json" and feed the last turn of
        # every match to load_summaries, which would treat a truncated meeting as a summary.
        record.finish(usage=usage, elapsed_time=time.time() - start_time, error=error)

        if discussion:
            partial_dir = save_dir / PARTIAL_MEETING_DIR_NAME
            save_meeting(save_dir=partial_dir, save_name=save_name, discussion=discussion)
            save_record(save_dir=partial_dir, save_name=save_name, record=record)
            print(f"Meeting failed. Partial discussion saved to {partial_dir / f'{save_name}.json'}")
        print("Usage before the failure:")
        usage.print_summary(elapsed_time=time.time() - start_time)
        raise

    record.finish(usage=usage, elapsed_time=time.time() - start_time, error=None)

    # Print the usage reported by the API, priced per model
    usage.print_summary(elapsed_time=time.time() - start_time)

    # Save the discussion as JSON and Markdown, plus the record of what produced it
    save_meeting(
        save_dir=save_dir,
        save_name=save_name,
        discussion=discussion,
    )
    save_record(save_dir=save_dir, save_name=save_name, record=record)

    if structured_output is not None:
        save_output(save_dir=save_dir, save_name=save_name, output=structured_output)

        return structured_output

    # Optionally, return summary
    if return_summary:
        return get_summary(discussion)

    return None
