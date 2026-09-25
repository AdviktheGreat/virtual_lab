"""Schemas for the decisions a meeting can reach.

These are the shapes that let one meeting's conclusion become the next meeting's input without a
person in between. Every field is required and none has a default, because the structured output
mode the API enforces requires it.
"""

from collections.abc import Iterable

from pydantic import BaseModel, Field, model_validator

from virtual_lab.agent import Agent


class AgentSpec(BaseModel):
    """A scientist a meeting decided to bring onto the project.

    The field wording matters: each one completes a sentence in the agent's system prompt, so the
    descriptions tell the model what grammatical form to produce.
    """

    title: str = Field(description="The scientist's job title, for example 'Immunologist'.")
    expertise: str = Field(
        description="What the scientist is an expert in, phrased to complete the sentence "
        "'Your expertise is in ...', for example 'antibody engineering and immunogenicity'."
    )
    goal: str = Field(
        description="What the scientist aims to achieve, phrased to complete the sentence "
        "'Your goal is to ...', for example 'design nanobodies with broad neutralising activity'."
    )
    role: str = Field(
        description="What the scientist will do on this project, phrased to complete the sentence "
        "'Your role is to ...', for example 'advise on the immunogenicity of proposed designs'."
    )

    def to_agent(self, model: str) -> Agent:
        """Builds the agent this spec describes.

        :param model: The model the agent will use.
        :return: The agent.
        """
        return Agent(
            title=self.title,
            expertise=self.expertise,
            goal=self.goal,
            role=self.role,
            model=model,
        )


class TeamRoster(BaseModel):
    """The team a meeting decided to assemble."""

    team_members: list[AgentSpec] = Field(
        description="The scientists to invite, excluding the principal investigator."
    )

    @model_validator(mode="after")
    def check_titles_are_unique(self) -> "TeamRoster":
        """Rejects a roster with repeated titles.

        Titles identify who is speaking, both in the transcript and in the author names sent to
        the API, and run_meeting refuses a team with duplicates. Catching it here points at the
        roster that caused it rather than at the meeting that failed later.
        """
        titles = [member.title for member in self.team_members]
        duplicates = sorted({title for title in titles if titles.count(title) > 1})

        if duplicates:
            raise ValueError(f"Team member titles must be unique; repeated: {', '.join(duplicates)}")

        return self

    def to_agents(self, model: str) -> tuple[Agent, ...]:
        """Builds the agents this roster describes.

        :param model: The model every agent on the team will use.
        :return: The agents, in the order the meeting listed them.
        """
        return tuple(member.to_agent(model=model) for member in self.team_members)


def normalize_title(title: str) -> str:
    """Reduces a title to a form that can be compared across the way different turns write it.

    :param title: The title to normalize.
    :return: The title stripped of surrounding whitespace and case.
    """
    return " ".join(title.split()).casefold()


class ComponentAssignment(BaseModel):
    """One piece of work, and who a meeting decided should do it."""

    component: str = Field(
        description="The piece of work to be built, for example 'ESM' or 'AlphaFold-Multimer'."
    )
    assignee_title: str = Field(
        description="The exact title of the team member who will build it, as written in the "
        "discussion. It must be one of the team members present in this meeting."
    )
    rationale: str = Field(description="Why this team member was chosen for this component.")


class ImplementationPlan(BaseModel):
    """Who will build each part of the work."""

    assignments: list[ComponentAssignment] = Field(
        description="One entry per component. A team member may appear more than once, but a "
        "component may not."
    )

    @model_validator(mode="after")
    def check_components_are_unique(self) -> "ImplementationPlan":
        """Rejects a plan that assigns the same component twice, which has no single owner."""
        components = [normalize_title(assignment.component) for assignment in self.assignments]
        duplicates = sorted({name for name in components if components.count(name) > 1})

        if duplicates:
            raise ValueError(f"Each component may be assigned once; repeated: {', '.join(duplicates)}")

        return self

    def resolve(self, team: Iterable[Agent]) -> dict[str, Agent]:
        """Matches each assignment to an agent, so the plan can decide who runs the next meeting.

        Resolving against the real team is the point: a title the meeting invented, or a member
        who was never present, has to fail here rather than silently assigning the work to nobody.

        :param team: The agents the work can be assigned to.
        :raises ValueError: If an assignee is not among the given agents.
        :return: A mapping from component to the agent who will build it, in assignment order.
        """
        title_to_agent = {normalize_title(agent.title): agent for agent in team}

        resolved: dict[str, Agent] = {}

        for assignment in self.assignments:
            agent = title_to_agent.get(normalize_title(assignment.assignee_title))

            if agent is None:
                available = ", ".join(sorted(agent.title for agent in title_to_agent.values()))
                raise ValueError(
                    f'Component "{assignment.component}" was assigned to '
                    f'"{assignment.assignee_title}", who is not on the team. '
                    f"Team members: {available or 'none'}."
                )

            resolved[assignment.component] = agent

        return resolved
