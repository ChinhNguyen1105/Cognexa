from cognexa.assistant.prompt import PromptBuilder
import pytest

from cognexa.assistant.prompt import PromptBuilder

@pytest.fixture
def prompt_builder():
    return PromptBuilder(
        system_prompt="You are Cognexa AI Assistant."
    )


def test_build_prompt_without_context(prompt_builder):
    messages = prompt_builder.build(
        history=[],
        user_message="Hello",
    )

    assert messages == [
        {
            "role": "system",
            "content": "You are Cognexa AI Assistant.",
        },
        {
            "role": "user",
            "content": "Hello",
        },
    ]


def test_build_prompt_with_context(prompt_builder):
    context = "Python is a programming language."

    messages = prompt_builder.build(
        history=[],
        user_message="What is Python?",
        context=context,
    )

    system_message = messages[0]["content"]

    assert context in system_message
    assert "Retrieved context:" in system_message
    assert "clearly state" in system_message


def test_build_prompt_preserves_conversation_history(prompt_builder):
    history = [
        {
            "role": "user",
            "content": "Hello",
        },
        {
            "role": "assistant",
            "content": "Hello! How can I help you?",
        },
    ]

    messages = prompt_builder.build(
        history=history,
        user_message="What is Python?",
        context="Python is a programming language.",
    )

    assert messages[1:3] == history
    assert messages[3] == {
        "role": "user",
        "content": "What is Python?",
    }


def test_build_prompt_without_context_preserves_system_prompt(
    prompt_builder,
):
    messages = prompt_builder.build(
        history=[],
        user_message="Hello",
        context=None,
    )

    assert messages[0]["content"] == (
        "You are Cognexa AI Assistant."
    )


def test_build_prompt_with_empty_context(prompt_builder):
    messages = prompt_builder.build(
        history=[],
        user_message="Hello",
        context="",
    )

    assert messages[0]["content"] == (
        "You are Cognexa AI Assistant."
    )


def test_build_prompt_does_not_modify_original_history(prompt_builder):
    history = [
        {
            "role": "user",
            "content": "Previous question",
        }
    ]

    original_history = history.copy()

    prompt_builder.build(
        history=history,
        user_message="New question",
        context="Some knowledge.",
    )

    assert history == original_history


