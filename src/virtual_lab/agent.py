"""The LLM agent class."""

import re

from openai.types.chat import ChatCompletionMessageParam

from virtual_lab.constants import MAX_AGENT_NAME_LENGTH


class Agent:
    """An LLM agent."""

    def __init__(self, title: str, expertise: str, goal: str, role: str, model: str) -> None:
        """Initializes the agent.

        :param title: The title of the agent.
        :param expertise: The expertise of the agent.
        :param goal: The goal of the agent.
        :param role: The role of the agent.
        """
        self.title = title
        self.expertise = expertise
        self.goal = goal
        self.role = role
        self.model = model

    @property
    def prompt(self) -> str:
        """Returns the prompt for the agent."""
        return (
            f"You are a {self.title}. "
            f"Your expertise is in {self.expertise}. "
            f"Your goal is to {self.goal}. "
            f"Your role is to {self.role}."
        )

    @property
    def name(self) -> str:
        """Returns the agent's title as an author name the API will accept.

        The API restricts message author names to letters, digits, underscores, and hyphens, so
        a title such as "Principal Investigator" becomes "Principal_Investigator". The title is
        still what appears in the saved transcript.
        """
        name = re.sub(r"[^a-zA-Z0-9_-]+", "_", self.title).strip("_")

        # A title made entirely of punctuation would otherwise produce an empty, invalid name
        return name[:MAX_AGENT_NAME_LENGTH] or "agent"

    def with_model(self, model: str) -> "Agent":
        """Returns a copy of the agent that uses a different model.

        :param model: The model for the new agent.
        :return: A copy of the agent using the given model.
        """
        return Agent(
            title=self.title,
            expertise=self.expertise,
            goal=self.goal,
            role=self.role,
            model=model,
        )

    @property
    def message(self) -> ChatCompletionMessageParam:
        """Returns the message for the agent in OpenAI API form."""
        return {
            "role": "system",
            "content": self.prompt,
        }

    def __hash__(self) -> int:
        """Returns the hash of the agent."""
        return hash(self.title)

    def __eq__(self, other: object) -> bool:
        """Checks if the agent is equal to another agent (based on title)."""
        if not isinstance(other, Agent):
            return False

        return (
            self.title == other.title
            and self.expertise == other.expertise
            and self.goal == other.goal
            and self.role == other.role
            and self.model == other.model
        )

    def __str__(self) -> str:
        """Returns the string representation of the agent (i.e., the agent's title)."""
        return self.title

    def __repr__(self) -> str:
        """Returns the string representation of the agent (i.e., the agent's title)."""
        return self.title
