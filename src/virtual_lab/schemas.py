"""Schemas for the decisions a meeting can reach.

These are the shapes that let one meeting's conclusion become the next meeting's input without a
person in between. Every field is required and none has a default, because the structured output
mode the API enforces requires it.
"""

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
