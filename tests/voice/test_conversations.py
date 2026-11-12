"""Conversations with Iris in a virtual room: python -m pytest -m voice

Each conversation of irisdev.voicetest is held with Iris and each turn
checked: heard as said, as one turn, answered aloud, in time, with the tools
it should use. A scripted model answers and a scripted web is searched, so
this tests the voice and the tool loop; with IRIS_VOICETEST_HOST and
IRIS_VOICETEST_MODEL a real model does, and its replies are checked too.
IRIS_VOICETEST_BUDGETS (voice_pipeline=3.5,...) sets other budgets.

- voice: the default groups (everyday, search, turn-taking, interruptions,
  hearing, conversation, performance);
- voice_slow: Whisper's sizes and a session of 60 turns;
- voice_network: web search over the real network (IRIS_VOICETEST_WEB=real).

A check known to fail today (Conversation.expected_failures) makes the test
an expected failure, not a failure; one that passes is reported.
"""

import os
from pathlib import Path

import pytest

from irisdev.voicetest import IrisUnderTest, ScriptedConversations, Space, StubModel, UserVoice
from irisdev.voicetest.conversations import DEFAULT_GROUPS, group
from irisdev.voicetest.resources import Resources
from irisdev.voicetest.scripted import budgets_from
from irisdev.voicetest.stub_llm import MODEL


@pytest.fixture(scope="module")
def iris(tmp_path_factory):
    """One Iris for all conversations; each conversation puts back what it
    changed."""
    host, model = os.environ.get("IRIS_VOICETEST_HOST"), os.environ.get("IRIS_VOICETEST_MODEL")
    stub = None if host else StubModel()
    space = Space()
    under_test = IrisUnderTest(space.endpoint("iris"), host or stub.url, model or MODEL, stub=stub)
    under_test.start()
    yield under_test, space, stub
    under_test.close()
    if stub:
        stub.close()


def hold(iris, conversation):
    assistant, space, stub = iris
    runner = ScriptedConversations(
        assistant,
        space,
        UserVoice(Path.home() / ".cache" / "iris-voicetest"),
        check_replies=stub is None,
        reply_timeout=300.0 if stub is None else 180.0,
        budgets=budgets_from(os.environ.get("IRIS_VOICETEST_BUDGETS", "")),
        report=lambda line: None,
        stub=stub,
        resources=Resources(),
    )
    results = runner.hold(conversation)
    skipped = [r.skipped for r in results if r.skipped and not r.turn]
    if skipped:
        pytest.skip(skipped[0])
    failures = [f"turn {r.turn}: {failure}" for r in results for failure in r.failures]
    assert not failures, "\n".join(failures)
    expected = [f"turn {r.turn}: {e}" for r in results for e in r.expected]
    if expected:
        pytest.xfail("known today:\n" + "\n".join(expected))


@pytest.mark.voice
@pytest.mark.parametrize("conversation", group(*DEFAULT_GROUPS), ids=lambda c: c.name)
def test_conversation(iris, conversation):
    hold(iris, conversation)


@pytest.mark.voice_slow
@pytest.mark.parametrize("conversation", group("slow"), ids=lambda c: c.name)
def test_long_run(iris, conversation):
    hold(iris, conversation)


@pytest.mark.voice_network
@pytest.mark.parametrize("conversation", group("network"), ids=lambda c: c.name)
def test_real_web(iris, conversation):
    hold(iris, conversation)
