"""Tests for running a meeting's code and giving its author the failures back.

The executor here is real and the code really runs; only the model is faked. That way the loop
is tested against actual tracebacks and exit codes rather than against a mock's idea of them.
"""

import json

import pytest

from virtual_lab.agent import Agent
from virtual_lab.artifacts import CodeArtifacts, CodeFile
from virtual_lab.constants import EXECUTION_DIR_NAME
from virtual_lab.execution import DockerExecutor, LocalExecutor
from virtual_lab.prompts import code_repair_prompt
from virtual_lab.run_meeting import run_meeting
from virtual_lab.repair import (
    RepairAttempt,
    error_signature,
    merge_artifacts,
    run_with_repair,
    save_execution_record,
)

from conftest import FakeClient, parsed_response

DOCKER = DockerExecutor()

needs_docker = pytest.mark.skipif(
    not DOCKER.is_available(), reason="Docker daemon is not reachable"
)


def script(contents: str, filename: str = "analysis.py") -> CodeArtifacts:
    """Builds a single-file artifact set containing the given python source."""
    return CodeArtifacts(
        files=[
            CodeFile(
                filename=filename,
                language="python",
                description="Does the analysis.",
                contents=contents,
            )
        ]
    )


WORKING = script("print('the answer is 42')")
BROKEN = script("raise ValueError('boom')")


def local() -> LocalExecutor:
    return LocalExecutor(warn=False)


def run(client: FakeClient, artifacts: CodeArtifacts, author: Agent, tmp_path, **kwargs):
    return run_with_repair(
        artifacts=artifacts,
        author=author,
        save_dir=tmp_path,
        executor=local(),
        client=client,
        **kwargs,
    )


class TestCodeThatWorks:
    def test_it_runs_once_and_succeeds(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        outcome = run(fake_client, WORKING, team_member, tmp_path)

        assert outcome.succeeded
        assert outcome.num_attempts == 1
        assert not outcome.was_repaired

    def test_no_repair_is_requested(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        # A meeting whose code works must not pay for a correction
        run(fake_client, WORKING, team_member, tmp_path)

        assert fake_client.completions.parse_calls == []

    def test_the_output_is_captured(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        outcome = run(fake_client, WORKING, team_member, tmp_path)

        assert "the answer is 42" in outcome.attempts[0].results[0][1].stdout

    def test_the_files_are_on_disk(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        outcome = run(fake_client, WORKING, team_member, tmp_path)

        assert outcome.paths[0].read_text() == "print('the answer is 42')"


class TestCodeThatIsRepaired:
    def test_a_failure_is_fixed_and_rerun(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [
            parsed_response(parsed=script("print('fixed')"))
        ]

        outcome = run(fake_client, BROKEN, team_member, tmp_path)

        assert outcome.succeeded
        assert outcome.num_attempts == 2
        assert outcome.was_repaired

    def test_the_repaired_code_is_what_ends_up_on_disk(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [
            parsed_response(parsed=script("print('fixed')"))
        ]

        outcome = run(fake_client, BROKEN, team_member, tmp_path)

        assert outcome.paths[0].read_text() == "print('fixed')"
        assert outcome.artifacts.files[0].contents == "print('fixed')"

    def test_exactly_one_repair_is_requested(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [
            parsed_response(parsed=script("print('fixed')"))
        ]

        run(fake_client, BROKEN, team_member, tmp_path)

        assert len(fake_client.completions.parse_calls) == 1

    def test_the_author_is_asked_not_the_critic(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        # The author holds the reasoning behind the code; debugging is not the critic's job
        fake_client.completions.parsed_responses = [
            parsed_response(parsed=script("print('fixed')"))
        ]

        run(fake_client, BROKEN, team_member, tmp_path)
        messages = fake_client.completions.parse_calls[0]["messages"]

        assert messages[0]["role"] == "system"
        assert team_member.title in messages[0]["content"]

    def test_the_request_carries_the_traceback(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [
            parsed_response(parsed=script("print('fixed')"))
        ]

        run(fake_client, BROKEN, team_member, tmp_path)
        request = fake_client.completions.parse_calls[0]["messages"][1]["content"]

        assert "ValueError" in request
        assert "boom" in request

    def test_the_request_carries_the_current_code(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        # The repair is a fresh call, so the code has to be restated or there is nothing to fix
        fake_client.completions.parsed_responses = [
            parsed_response(parsed=script("print('fixed')"))
        ]

        run(fake_client, BROKEN, team_member, tmp_path)
        request = fake_client.completions.parse_calls[0]["messages"][1]["content"]

        assert "raise ValueError('boom')" in request
        assert "analysis.py" in request

    def test_a_repair_is_asked_for_as_a_schema(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [
            parsed_response(parsed=script("print('fixed')"))
        ]

        run(fake_client, BROKEN, team_member, tmp_path)

        assert fake_client.completions.parse_calls[0]["response_format"] is CodeArtifacts


class TestGivingUp:
    def test_the_same_error_twice_stops_the_loop(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        # An agent that reproduces its own error will keep reproducing it, and each further
        # attempt costs another round of code generation
        fake_client.completions.parsed_responses = [
            parsed_response(parsed=script("raise ValueError('boom')"))
        ]

        outcome = run(fake_client, BROKEN, team_member, tmp_path, max_attempts=5)

        assert not outcome.succeeded
        assert outcome.stopped_early
        assert outcome.num_attempts == 2
        assert len(fake_client.completions.parse_calls) == 1

    def test_different_errors_use_the_whole_budget(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [
            parsed_response(parsed=script("raise ValueError('second')")),
            parsed_response(parsed=script("raise ValueError('third')")),
        ]

        outcome = run(fake_client, BROKEN, team_member, tmp_path, max_attempts=3)

        assert not outcome.succeeded
        assert not outcome.stopped_early
        assert outcome.num_attempts == 3
        assert len(fake_client.completions.parse_calls) == 2

    def test_the_last_attempt_does_not_pay_for_a_correction(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        # Asking for a fix nobody will run is money spent for nothing
        fake_client.completions.parsed_responses = [
            parsed_response(parsed=script("raise ValueError('second')")),
            parsed_response(parsed=script("raise ValueError('third')")),
        ]

        run(fake_client, BROKEN, team_member, tmp_path, max_attempts=3)

        assert len(fake_client.completions.parse_calls) == 2

    def test_a_single_attempt_never_asks_for_a_repair(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        outcome = run(fake_client, BROKEN, team_member, tmp_path, max_attempts=1)

        assert not outcome.succeeded
        assert outcome.num_attempts == 1
        assert fake_client.completions.parse_calls == []

    def test_zero_attempts_is_refused(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            run(fake_client, BROKEN, team_member, tmp_path, max_attempts=0)

    def test_a_timeout_is_treated_as_a_failure_to_repair(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [
            parsed_response(parsed=script("print('fixed')"))
        ]

        outcome = run(
            fake_client,
            script("while True: pass"),
            team_member,
            tmp_path,
            max_attempts=2,
            timeout=2,
        )

        assert outcome.attempts[0].results[0][1].timed_out
        assert outcome.succeeded
        assert len(fake_client.completions.parse_calls) == 1


class TestWhatTheCriticIsTold:
    def test_success_on_the_first_try_is_stated_plainly(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        outcome = run(fake_client, WORKING, team_member, tmp_path)

        assert "ran successfully on the first attempt" in outcome.report()
        assert "the answer is 42" in outcome.report()

    def test_a_repair_is_disclosed_rather_than_hidden(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        # A critic that is not told the code had to be corrected cannot weigh that
        fake_client.completions.parsed_responses = [
            parsed_response(parsed=script("print('fixed')"))
        ]

        outcome = run(fake_client, BROKEN, team_member, tmp_path)

        assert "after 2 attempts" in outcome.report()
        assert "corrected" in outcome.report()

    def test_abandonment_is_disclosed(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [
            parsed_response(parsed=script("raise ValueError('boom')"))
        ]

        outcome = run(fake_client, BROKEN, team_member, tmp_path, max_attempts=5)
        report = outcome.report()

        assert "same error recurred" in report
        assert "ValueError" in report

    def test_total_failure_is_disclosed(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        outcome = run(fake_client, BROKEN, team_member, tmp_path, max_attempts=1)

        assert "failed on all 1 attempt" in outcome.report()

    def test_a_long_traceback_is_shortened(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        noisy = script("import sys\nsys.stderr.write('x' * 100000)\nraise ValueError('end')")

        outcome = run(fake_client, noisy, team_member, tmp_path, max_attempts=1)

        assert len(outcome.report()) < 20_000


class TestRepairPrompt:
    def test_it_forbids_stubbing_out_the_computation(self, team_member: Agent) -> None:
        # This is how a random-number stub came to stand in for a real implementation
        prompt = code_repair_prompt(
            agent=team_member,
            files=WORKING.files,
            filename="analysis.py",
            report="It failed.",
            attempt=1,
            max_attempts=3,
        )

        assert "do not remove or stub out the computation" in prompt.lower()
        assert "placeholder, random, or hard-coded values" in prompt
        assert "catch the exception in order to continue past it" in prompt

    def test_it_asks_for_whole_files_not_diffs(self, team_member: Agent) -> None:
        prompt = code_repair_prompt(
            agent=team_member,
            files=WORKING.files,
            filename="analysis.py",
            report="It failed.",
            attempt=1,
            max_attempts=3,
        )

        assert "Return every file in full" in prompt
        assert "Do not return diffs" in prompt

    def test_it_states_the_remaining_budget(self, team_member: Agent) -> None:
        prompt = code_repair_prompt(
            agent=team_member,
            files=WORKING.files,
            filename="analysis.py",
            report="It failed.",
            attempt=2,
            max_attempts=3,
        )

        assert "attempt 2 of 3" in prompt


class TestMergingARepair:
    def test_an_unmentioned_file_survives(self) -> None:
        # A model that returns only the file it changed must not thereby delete the rest
        previous = CodeArtifacts(
            files=[*script("broken", "main.py").files, *script("helper", "util.py").files]
        )
        merged = merge_artifacts(previous=previous, repaired=script("fixed", "main.py"))

        assert [file.filename for file in merged.files] == ["main.py", "util.py"]
        assert merged.files[0].contents == "fixed"
        assert merged.files[1].contents == "helper"

    def test_a_new_file_is_added(self) -> None:
        merged = merge_artifacts(
            previous=script("broken", "main.py"),
            repaired=CodeArtifacts(
                files=[*script("fixed", "main.py").files, *script("new", "extra.py").files]
            ),
        )

        assert [file.filename for file in merged.files] == ["main.py", "extra.py"]

    def test_order_is_preserved(self) -> None:
        previous = CodeArtifacts(
            files=[
                *script("a", "a.py").files,
                *script("b", "b.py").files,
                *script("c", "c.py").files,
            ]
        )
        merged = merge_artifacts(previous=previous, repaired=script("fixed", "b.py"))

        assert [file.filename for file in merged.files] == ["a.py", "b.py", "c.py"]


class TestErrorSignature:
    def test_the_same_exception_matches(self) -> None:
        from virtual_lab.execution import ExecutionResult

        first = ExecutionResult(
            command=(), exit_code=1, stdout="", stderr="Traceback\nValueError: boom", duration=1
        )
        second = ExecutionResult(
            command=(), exit_code=1, stdout="other", stderr="Different\nValueError: boom", duration=2
        )

        assert error_signature(first) == error_signature(second)

    def test_a_different_message_does_not_match(self) -> None:
        from virtual_lab.execution import ExecutionResult

        first = ExecutionResult(command=(), exit_code=1, stdout="", stderr="ValueError: a", duration=1)
        second = ExecutionResult(command=(), exit_code=1, stdout="", stderr="ValueError: b", duration=1)

        assert error_signature(first) != error_signature(second)


class TestExecutionRecord:
    def test_every_attempt_is_recorded(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        # The cost of repairs has to be visible rather than buried
        fake_client.completions.parsed_responses = [
            parsed_response(parsed=script("print('fixed')"))
        ]
        outcome = run(fake_client, BROKEN, team_member, tmp_path)

        path = save_execution_record(
            save_dir=tmp_path,
            save_name="discussion",
            outcome=outcome,
            author=team_member,
            executor=local(),
        )
        record = json.loads(path.read_text())

        assert record["num_attempts"] == 2
        assert record["was_repaired"] is True
        assert len(record["attempts"]) == 2
        assert record["attempts"][0]["succeeded"] is False
        assert record["attempts"][1]["succeeded"] is True

    def test_the_repair_cost_is_recorded(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [
            parsed_response(parsed=script("print('fixed')"))
        ]
        outcome = run(fake_client, BROKEN, team_member, tmp_path)

        assert outcome.usage.num_calls == 1
        assert outcome.usage.input_tokens > 0

    def test_the_record_says_whether_it_was_sandboxed(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        outcome = run(fake_client, WORKING, team_member, tmp_path)

        path = save_execution_record(
            save_dir=tmp_path,
            save_name="discussion",
            outcome=outcome,
            author=team_member,
            executor=local(),
        )
        record = json.loads(path.read_text())

        assert record["executor"]["type"] == "LocalExecutor"
        assert record["attempts"][0]["runs"][0]["sandboxed"] is False

    def test_a_sandboxed_run_records_its_limits(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        outcome = run(fake_client, WORKING, team_member, tmp_path)

        path = save_execution_record(
            save_dir=tmp_path,
            save_name="discussion",
            outcome=outcome,
            author=team_member,
            executor=DockerExecutor(),
        )
        record = json.loads(path.read_text())

        assert record["executor"]["image"] == "python:3.12-slim"
        assert record["executor"]["allow_network"] is False
        assert record["executor"]["memory_limit"] == "2g"

    def test_the_record_lives_beside_the_others_not_among_them(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        outcome = run(fake_client, WORKING, team_member, tmp_path)
        save_execution_record(
            save_dir=tmp_path,
            save_name="discussion_1",
            outcome=outcome,
            author=team_member,
            executor=local(),
        )

        assert (tmp_path / EXECUTION_DIR_NAME / "discussion_1.json").exists()
        assert list(tmp_path.glob("discussion_*.json")) == []

    def test_the_final_code_is_recorded(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [
            parsed_response(parsed=script("print('fixed')"))
        ]
        outcome = run(fake_client, BROKEN, team_member, tmp_path)

        path = save_execution_record(
            save_dir=tmp_path,
            save_name="discussion",
            outcome=outcome,
            author=team_member,
            executor=local(),
        )
        record = json.loads(path.read_text())

        assert record["final_files"][0]["contents"] == "print('fixed')"
        assert record["author"]["title"] == "Immunologist"


class TestNothingToRun:
    def test_documentation_only_output_is_not_a_failure(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        notes = CodeArtifacts(
            files=[
                CodeFile(
                    filename="notes.md", language="text", description="Notes.", contents="# Notes"
                )
            ]
        )

        outcome = run(fake_client, notes, team_member, tmp_path)

        assert outcome.succeeded
        assert outcome.num_attempts == 1
        assert outcome.attempts[0].results == ()
        assert fake_client.completions.parse_calls == []


class TestAttemptHelpers:
    def test_an_attempt_with_nothing_run_counts_as_succeeded(self) -> None:
        assert RepairAttempt(index=1, results=()).succeeded
        assert RepairAttempt(index=1, results=()).failure is None


class TestResultsReachTheCritic:
    """The point of running code: a review of evidence rather than of intentions."""

    def test_the_real_output_is_put_in_front_of_the_reviewer(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        outcome = run(fake_client, WORKING, team_member, tmp_path)

        run_meeting(
            meeting_type="individual",
            agenda="Review the results of running the analysis.",
            save_dir=tmp_path,
            save_name="review",
            team_member=team_member,
            contexts=(outcome.report(),),
            num_rounds=0,
        )
        start_prompt = fake_client.completions.calls[0]["messages"][1]["content"]

        assert "the answer is 42" in start_prompt
        assert "ran successfully on the first attempt" in start_prompt

    def test_a_reviewer_is_told_when_the_code_never_worked(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        # Without this, a review of failed work reads exactly like a review of working work
        outcome = run(fake_client, BROKEN, team_member, tmp_path, max_attempts=1)

        run_meeting(
            meeting_type="individual",
            agenda="Review the results of running the analysis.",
            save_dir=tmp_path,
            save_name="review",
            team_member=team_member,
            contexts=(outcome.report(),),
            num_rounds=0,
        )
        start_prompt = fake_client.completions.calls[0]["messages"][1]["content"]

        assert "failed" in start_prompt
        assert "ValueError" in start_prompt


class TestRepairInTheSandbox:
    @needs_docker
    def test_the_whole_loop_works_in_a_container(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [
            parsed_response(parsed=script("print('fixed in the sandbox')"))
        ]

        outcome = run_with_repair(
            artifacts=BROKEN,
            author=team_member,
            save_dir=tmp_path,
            executor=DOCKER,
            client=fake_client,
        )

        assert outcome.succeeded
        assert outcome.num_attempts == 2
        assert "fixed in the sandbox" in outcome.attempts[1].results[0][1].stdout
        assert outcome.attempts[0].results[0][1].sandboxed is True

    @needs_docker
    def test_a_real_traceback_from_the_container_reaches_the_author(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [
            parsed_response(parsed=script("print('fixed')"))
        ]

        run_with_repair(
            artifacts=script("import nonexistent_package_xyz"),
            author=team_member,
            save_dir=tmp_path,
            executor=DOCKER,
            client=fake_client,
        )
        request = fake_client.completions.parse_calls[0]["messages"][1]["content"]

        assert "ModuleNotFoundError" in request
        assert "nonexistent_package_xyz" in request
