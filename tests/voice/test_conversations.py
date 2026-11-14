"""Conversations with Iris in a virtual room: python -m pytest -m voice

Each conversation of irisdev.voicetest is held with Iris and each turn
checked: heard as said, as one turn, answered aloud, in time. A scripted
model answers, so this tests the voice; with IRIS_VOICETEST_HOST and
IRIS_VOICETEST_MODEL a real model does, and its replies are checked too.
IRIS_VOICETEST_BUDGETS (voice_pipeline=3.5,...) sets other budgets.
"""

import os
from pathlib import Path

import pytest

from irisdev.voicetest import CONVERSATIONS, IrisUnderTest, ScriptedConversations, Space, StubModel, UserVoice
from irisdev.voicetest.scripted import budgets_from
from irisdev.voicetest.stub_llm import MODEL

pytestmark = pytest.mark.voice


@pytest.fixture(scope="module")
def iris(tmp_path_factory):
    host, model = os.environ.get("IRIS_VOICETEST_HOST"), os.environ.get("IRIS_VOICETEST_MODEL")
    stub = None if host else StubModel()
    space = Space()
    under_test = IrisUnderTest(space.endpoint("iris"), host or stub.url, model or MODEL)
    under_test.start()
    yield under_test, space, stub is None
    under_test.close()
    if stub:
        stub.close()


@pytest.mark.parametrize("conversation", CONVERSATIONS, ids=lambda c: c.name)
def test_conversation(iris, conversation):
    assistant, space, real_model = iris
    runner = ScriptedConversations(
        assistant,
        space,
        UserVoice(Path.home() / ".cache" / "iris-voicetest"),
        check_replies=real_model,
        reply_timeout=300.0 if real_model else 180.0,
        budgets=budgets_from(os.environ.get("IRIS_VOICETEST_BUDGETS", "")),
        report=lambda line: None,
    )
    results = runner.hold(conversation)
    failures = [f"turn {r.turn}: {failure}" for r in results for failure in r.failures]
    assert not failures, "\n".join(failures)
