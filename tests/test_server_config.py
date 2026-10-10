"""Tests of what a conversation is set up with, and that it is carried on as it was set up."""

import json
from pathlib import Path
from typing import Any

import pytest

from virtual_lab.prompts import PRINCIPAL_INVESTIGATOR, SCIENTIFIC_CRITIC
from virtual_lab.scientists import SCIENTISTS
from virtual_lab.server.config import (
    CONFIG_FILE_NAME,
    DEFAULT_TEAM,
    PUBLIC_SETTINGS,
    ChatConfig,
    describe_models,
    describe_scientist,
    library,
    load_config,
    public_settings,
    resolve_config,
    save_config,
    update_settings,
)
from virtual_lab.ui.runs import CodeSetup
from virtual_lab.ui.workspace import Settings, Workspace

from conftest import TEST_MODEL

NANOBODY_EXPERT = {
    "title": "Nanobody Engineer",
    "expertise": "single-domain antibodies",
    "goal": "design binders",
    "role": "propose sequences",
}


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path / "workspace")


def settings(**changes: Any) -> Settings:
    return Settings(**{"model": TEST_MODEL, **changes})


def titles(members: Any) -> list[str]:
    return [member["title"] for member in members]


class TestResolving:
    def test_what_the_request_leaves_out_is_what_the_settings_and_the_library_say(self, workspace: Workspace) -> None:
        chosen = settings(
            max_cost=1.5, stream=False, code="local", sandbox="bio", network=False, python="/usr/bin/python3"
        )

        config = resolve_config({}, chosen, workspace)

        assert config.model == TEST_MODEL
        assert config.lead["title"] == PRINCIPAL_INVESTIGATOR.title
        assert titles(config.team) == list(DEFAULT_TEAM)
        assert config.code == CodeSetup(
            where="local", sandbox="bio", network=False, python="/usr/bin/python3", resources="retrieve"
        )
        assert config.max_cost == 1.5
        assert config.stream is False
        assert config.commercial_mode is False

    def test_what_the_request_gives_is_what_it_is_set_up_with(self, workspace: Workspace) -> None:
        config = resolve_config(
            {
                "model": "gpt-5.2",
                "team": ["Scientific Critic"],
                "code": {"where": "docker", "resources": "all"},
                "max_cost": 0.25,
                "stream": False,
                "commercial_mode": True,
            },
            settings(code="none", sandbox="full", stream=True),
            workspace,
        )

        assert config.model == "gpt-5.2"
        assert titles(config.team) == ["Scientific Critic"]
        # What the request names of the code is changed, and the rest is the settings'
        assert config.code == CodeSetup(where="docker", sandbox="full", network=True, python="", resources="all")
        assert config.max_cost == 0.25
        assert config.stream is False
        assert config.commercial_mode is True

    def test_no_limit_on_what_is_spent_is_asked_for_by_null_and_not_by_leaving_it_out(
        self, workspace: Workspace
    ) -> None:
        assert resolve_config({"max_cost": None}, settings(max_cost=2.0), workspace).max_cost is None
        assert resolve_config({}, settings(max_cost=2.0), workspace).max_cost == 2.0
        assert resolve_config({}, settings(max_cost=None), workspace).max_cost is None

    def test_a_lead_with_no_team_works_alone(self, workspace: Workspace) -> None:
        config = resolve_config({"team": []}, settings(), workspace)

        assert config.team == ()
        assert resolve_config({"team": None}, settings(), workspace).team != ()

    def test_the_lead_and_the_team_are_chosen_by_title_whatever_its_case_or_spacing(self, workspace: Workspace) -> None:
        config = resolve_config(
            {"lead": "  principal INVESTIGATOR ", "team": ["scientific critic", "Immunologist"]},
            settings(),
            workspace,
        )

        assert config.lead["title"] == PRINCIPAL_INVESTIGATOR.title
        assert titles(config.team) == [SCIENTIFIC_CRITIC.title, "Immunologist"]
        assert config.lead["role"] == PRINCIPAL_INVESTIGATOR.role

    def test_a_scientist_may_be_described_instead_of_chosen(self, workspace: Workspace) -> None:
        described = {key: f"  {value}\n  indeed " for key, value in NANOBODY_EXPERT.items()}

        config = resolve_config({"lead": described, "team": [described | {"title": "Second"}]}, settings(), workspace)

        assert config.lead == {key: f"{value} indeed" for key, value in NANOBODY_EXPERT.items()}
        assert config.team[0] == {**config.lead, "title": "Second"}

    def test_a_scientist_the_person_described_stands_in_place_of_the_librarys_with_the_title(
        self, workspace: Workspace
    ) -> None:
        mine = {**NANOBODY_EXPERT, "title": "Immunologist"}
        workspace.save_scientist(mine)
        workspace.save_scientist(NANOBODY_EXPERT)

        config = resolve_config({"team": ["immunologist", "Nanobody Engineer"]}, settings(), workspace)

        assert config.team == (mine, NANOBODY_EXPERT)

    @pytest.mark.parametrize(
        ("request_", "message"),
        [
            ({"team": ["Alchemist"]}, 'There is no scientist titled "Alchemist"'),
            ({"lead": "Nobody"}, 'There is no scientist titled "Nobody"'),
            ({"lead": {"title": "Lead", "expertise": "x", "goal": "y"}}, "A scientist needs role"),
            ({"lead": {"title": "Lead", "expertise": "x", "goal": " ", "role": ""}}, "A scientist needs goal, role"),
            ({"team": ["Immunologist", "immunologist"]}, "must have different titles"),
            ({"team": ["Principal Investigator"]}, "must have different titles"),
            ({"model": "not-a-model-anyone-serves"}, "Unable to determine the source"),
            ({"code": {"where": "cloud"}}, "Code runs in one of"),
            ({"code": {"sandbox": "huge"}}, "The sandbox is one of"),
            ({"code": {"resources": "some"}}, "The resources are one of"),
            ({"code": {"gpu": True}}, "unexpected keyword"),
            ({"max_cost": -1}, "max_cost must be a finite amount"),
            ({"max_cost": float("nan")}, "max_cost must be a finite amount"),
            ({"max_cost": float("inf")}, "max_cost must be a finite amount"),
            ({"max_cost": "plenty"}, "could not convert string to float"),
        ],
    )
    def test_something_that_cannot_be_used_is_refused_and_says_what(
        self, workspace: Workspace, request_: dict[str, Any], message: str
    ) -> None:
        with pytest.raises((ValueError, TypeError), match=message):
            resolve_config(request_, settings(), workspace)

    def test_the_default_team_is_the_critic_and_then_everyone_but_the_lead(self) -> None:
        assert DEFAULT_TEAM[0] == SCIENTIFIC_CRITIC.title
        assert PRINCIPAL_INVESTIGATOR.title not in DEFAULT_TEAM
        assert len(set(DEFAULT_TEAM)) == len(DEFAULT_TEAM)
        assert set(DEFAULT_TEAM) | {PRINCIPAL_INVESTIGATOR.title} == {agent.title for agent in SCIENTISTS} | {
            SCIENTIFIC_CRITIC.title,
            PRINCIPAL_INVESTIGATOR.title,
        }

    def test_the_agents_are_given_the_model(self, workspace: Workspace) -> None:
        config = resolve_config({"model": "gpt-5.2", "team": ["Immunologist"]}, settings(), workspace)

        assert config.lead_agent().model == "gpt-5.2"
        assert config.lead_agent().title == PRINCIPAL_INVESTIGATOR.title
        assert [agent.model for agent in config.team_agents()] == ["gpt-5.2"]


class TestDescribing:
    def test_the_library_is_the_built_in_scientists_and_then_those_the_person_described(
        self, workspace: Workspace
    ) -> None:
        workspace.save_scientist(NANOBODY_EXPERT)
        workspace.save_scientist({**NANOBODY_EXPERT, "title": "Immunologist", "role": "my own role"})

        found = library(workspace)

        by_title = {item["title"]: item for item in found}
        assert len(found) == len(by_title)
        assert by_title["Nanobody Engineer"]["builtin"] is False
        assert by_title["Immunologist"]["builtin"] is False
        assert by_title["Immunologist"]["role"] == "my own role"
        assert by_title[PRINCIPAL_INVESTIGATOR.title]["builtin"] is True
        assert {key for key in by_title[PRINCIPAL_INVESTIGATOR.title]} == {
            "title",
            "expertise",
            "goal",
            "role",
            "builtin",
        }

    def test_a_scientist_is_described_by_the_parts_every_one_has(self) -> None:
        assert describe_scientist(NANOBODY_EXPERT | {"extra": "ignored", "builtin": False}, []) == NANOBODY_EXPERT

    def test_a_title_is_looked_up_in_the_scientists_given(self) -> None:
        assert describe_scientist("NANOBODY engineer", [{**NANOBODY_EXPERT, "builtin": True}]) == NANOBODY_EXPERT


class TestKeeping:
    def test_what_a_conversation_is_set_up_with_is_kept_whole_and_read_back_the_same(
        self, workspace: Workspace, tmp_path: Path
    ) -> None:
        config = resolve_config(
            {"team": ["Immunologist"], "code": {"where": "local", "python": "/opt/python"}, "max_cost": None},
            settings(),
            workspace,
        )
        directory = tmp_path / "chat"
        directory.mkdir()

        save_config(directory, config)

        assert load_config(directory) == config
        assert not list(directory.glob("*.tmp"))
        assert json.loads((directory / CONFIG_FILE_NAME).read_text())["code"]["python"] == "/opt/python"

    def test_a_conversation_that_was_set_not_to_stream_or_to_leave_out_what_is_not_for_commerce_is_read_back_so(
        self, workspace: Workspace, tmp_path: Path
    ) -> None:
        directory = tmp_path / "chat"
        directory.mkdir()
        for request in ({"stream": False, "commercial_mode": True}, {"stream": True, "commercial_mode": False}):
            config = resolve_config({"team": [], **request}, settings(), workspace)

            save_config(directory, config)
            loaded = load_config(directory)

            assert loaded is not None
            assert (loaded.stream, loaded.commercial_mode) == (request["stream"], request["commercial_mode"])

    def test_a_conversation_that_has_begun_is_not_changed_by_changing_the_library_or_the_settings(
        self, workspace: Workspace, tmp_path: Path
    ) -> None:
        config = resolve_config({"team": ["Immunologist"]}, settings(), workspace)
        directory = tmp_path / "chat"
        directory.mkdir()
        save_config(directory, config)
        was = config.team[0]

        workspace.save_scientist({**NANOBODY_EXPERT, "title": "Immunologist", "role": "something else entirely"})
        workspace.save_settings(settings(model="gpt-5.2", max_cost=99))

        loaded = load_config(directory)
        assert loaded is not None
        assert loaded.team[0] == was
        assert loaded.model == TEST_MODEL

    def test_nothing_is_read_from_a_conversation_without_a_setup_that_can_be_read(self, tmp_path: Path) -> None:
        directory = tmp_path / "chat"
        directory.mkdir()
        assert load_config(directory) is None

        good = ChatConfig(
            model=TEST_MODEL,
            lead=NANOBODY_EXPERT,
            team=(),
            code=CodeSetup(),
            max_cost=None,
            stream=True,
            commercial_mode=False,
        ).to_dict()
        broken: list[Any] = [
            "not json at all {",
            "[]",
            "null",
            json.dumps({key: value for key, value in good.items() if key != "model"}),
            json.dumps({**good, "lead": {"title": "only a title"}}),
            json.dumps({**good, "code": {"where": "cloud"}}),
            json.dumps({**good, "code": {"where": "none", "unknown": 1}}),
            json.dumps({**good, "team": None}),
            json.dumps({**good, "stream": None, "max_cost": None, "model": None, "code": None}),
        ]
        for text in broken:
            (directory / CONFIG_FILE_NAME).write_text(text)
            assert load_config(directory) is None, text

        (directory / CONFIG_FILE_NAME).write_text(json.dumps(good))
        assert load_config(directory) is not None


class TestSettings:
    def test_settings_are_shown_without_the_mcp_config_the_interface_keeps(self) -> None:
        shown = public_settings(Settings(mcp_config="/somewhere/servers.json"))

        assert tuple(shown) == PUBLIC_SETTINGS
        assert "mcp_config" not in shown

    def test_a_change_gives_settings_with_it_made_and_leaves_the_old_ones_as_they_were(self) -> None:
        old = Settings(max_cost=2.0, mcp_config="/keep/this.json")

        new = update_settings(old, {"model": "claude-sonnet-4-5", "max_cost": 3, "code": "docker", "sandbox": "bio"})

        assert (new.model, new.max_cost, new.code, new.sandbox) == ("claude-sonnet-4-5", 3.0, "docker", "bio")
        assert isinstance(new.max_cost, float)
        assert new.mcp_config == "/keep/this.json"
        assert (old.max_cost, old.model, old.code) == (2.0, Settings().model, "none")

    def test_a_limit_of_nothing_may_be_spent_can_be_set(self) -> None:
        assert update_settings(Settings(max_cost=2.0), {"max_cost": 0}).max_cost == 0.0

    def test_no_limit_can_be_set(self) -> None:
        assert update_settings(Settings(max_cost=2.0), {"max_cost": None}).max_cost is None

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"colour": "red"}, "There is no setting colour"),
            ({"mcp_config": "/x.json"}, "There is no setting mcp_config"),
            ({"model": ""}, "Unable to determine the source"),
            ({"model": "no-such-model"}, "Unable to determine the source"),
            ({"code": "cloud"}, "Code runs in one of"),
            ({"sandbox": "huge"}, "The sandbox is one of"),
            ({"max_cost": -0.01}, "max_cost must be a finite amount"),
            ({"max_cost": float("nan")}, "max_cost must be a finite amount"),
            ({"max_cost": float("inf")}, "max_cost must be a finite amount"),
        ],
    )
    def test_a_change_that_cannot_be_used_is_refused(self, changes: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            update_settings(Settings(), changes)


class TestModels:
    def test_the_models_say_whether_the_keys_they_need_are_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_SOURCE", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "set")

        found = {item["id"]: item for item in describe_models()}

        assert found["gpt-5.2"] == {"id": "gpt-5.2", "provider": "OpenAI", "ready": False}
        claude = next(item for item in found.values() if item["provider"] == "Anthropic")
        assert claude["ready"] is True

        monkeypatch.setenv("OPENAI_API_KEY", "set")
        assert {item["id"]: item for item in describe_models()}["gpt-5.2"]["ready"] is True

    def test_the_default_model_comes_first(self) -> None:
        assert describe_models()[0]["id"] == Settings().model
