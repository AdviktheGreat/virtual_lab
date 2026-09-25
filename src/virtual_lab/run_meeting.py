"""Runs a meeting with LLM agents."""

import time
from pathlib import Path
from typing import Literal

from openai import OpenAI, NOT_GIVEN
from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionMessageParam, ChatCompletionToolParam
from tqdm import trange, tqdm

from virtual_lab.agent import Agent
from virtual_lab.constants import (
    CONSISTENT_TEMPERATURE,
    DEFAULT_MAX_RETRIES,
    PARTIAL_MEETING_DIR_NAME,
    PUBMED_TOOL_DESCRIPTION,
)
from virtual_lab.prompts import (
    individual_meeting_agent_prompt,
    individual_meeting_critic_prompt,
    individual_meeting_start_prompt,
    SCIENTIFIC_CRITIC,
    team_meeting_start_prompt,
    team_meeting_team_lead_initial_prompt,
    team_meeting_team_lead_intermediate_prompt,
    team_meeting_team_lead_final_prompt,
    team_meeting_team_member_prompt,
)
from virtual_lab.utils import (
    MeetingUsage,
    check_context_length,
    get_max_input_tokens,
    get_summary,
    run_tools,
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
    return_summary: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> str | None:
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
    :param pubmed_search: Whether to include a PubMed search tool.
    :param return_summary: Whether to return the summary of the meeting.
    :param max_retries: The number of times to retry a failed API call, with exponential backoff.
    :raises Exception: If an API call fails after all retries. The completed portion of the
        discussion is saved under save_dir/partial/ before the error propagates.
    :return: The summary of the meeting (i.e., the last message) if return_summary is True, else None.
    """
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

    # Set up tools
    tools: list[ChatCompletionToolParam] | None = (
        [ChatCompletionToolParam(**PUBMED_TOOL_DESCRIPTION)] if pubmed_search else None  # type: ignore[misc]
    )

    # Track the token usage reported by the API, per model
    usage = MeetingUsage()

    # Warn only on the first request that approaches the model's input limit
    warned_about_context = False

    # Initialize discussion (list of agent/message dicts for output)
    discussion: list[dict[str, str]] = []

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

                # Call the chat completions API
                response = client.chat.completions.create(
                    model=agent.model,
                    messages=agent_messages,
                    temperature=temperature,
                    tools=tools if tools else NOT_GIVEN,
                )
                usage.add(model=agent.model, usage=response.usage)

                # Get the response message
                response_message = response.choices[0].message

                # Check if the model wants to call tools
                if response_message.tool_calls:
                    # Run the tools and get outputs
                    tool_outputs, tool_messages = run_tools(tool_calls=response_message.tool_calls)

                    # Add the assistant's message with tool_calls to the messages
                    assistant_tool_message: ChatCompletionAssistantMessageParam = {
                        "role": "assistant",
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

                    # Make another API call with tool results
                    agent_messages = [agent.message] + messages

                    # Tool output can be large, so re-check before sending it back
                    check_context_length(messages=agent_messages, model=agent.model)

                    response = client.chat.completions.create(
                        model=agent.model,
                        messages=agent_messages,
                        temperature=temperature,
                    )
                    usage.add(model=agent.model, usage=response.usage)
                    response_message = response.choices[0].message

                # Extract the response content
                response_content = response_message.content or ""

                # Add response to messages and discussion
                messages.append({"role": "assistant", "content": response_content})
                discussion.append({"agent": agent.title, "message": response_content})

                # If final round, only team lead or team member responds
                if round_index == num_rounds:
                    break
    except Exception:
        # A meeting can be many minutes and many dollars of work, so preserve whatever
        # completed. It goes in a subdirectory rather than alongside the finished meetings
        # because callers glob patterns like "discussion_*.json" and feed the last turn of
        # every match to load_summaries, which would treat a truncated meeting as a summary.
        if discussion:
            partial_dir = save_dir / PARTIAL_MEETING_DIR_NAME
            save_meeting(save_dir=partial_dir, save_name=save_name, discussion=discussion)
            print(f"Meeting failed. Partial discussion saved to {partial_dir / f'{save_name}.json'}")
        print("Usage before the failure:")
        usage.print_summary(elapsed_time=time.time() - start_time)
        raise

    # Print the usage reported by the API, priced per model
    usage.print_summary(elapsed_time=time.time() - start_time)

    # Save the discussion as JSON and Markdown
    save_meeting(
        save_dir=save_dir,
        save_name=save_name,
        discussion=discussion,
    )

    # Optionally, return summary
    if return_summary:
        return get_summary(discussion)

    return None
