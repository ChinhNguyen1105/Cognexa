from unittest.mock import MagicMock

import pytest

from cognexa.assistant.assistant import Assistant
from cognexa.assistant.conversation import Conversation


@pytest.fixture
def generator():
    return MagicMock()


@pytest.fixture
def prompt_builder():
    return MagicMock()


@pytest.fixture
def conversation():
    return MagicMock(spec=Conversation)


@pytest.fixture
def assistant(generator, prompt_builder, conversation):
    return Assistant(
        generator=generator,
        prompt_builder=prompt_builder,
        conversation=conversation,
    )


def create_document(
    document_id,
    content,
    metadata=None,
):
    return {
        "id": document_id,
        "content": content,
        "metadata": metadata or {},
        "distance": 0.5,
    }


class TestAssistantInitialization:

    def test_initializes_with_default_components(
        self, generator, prompt_builder
    ):
        result = Assistant(
            generator=generator,
            prompt_builder=prompt_builder,
        )

        assert result.generator is generator
        assert result.prompt_builder is prompt_builder
        assert isinstance(result.conversation, Conversation)
        assert result.knowledge_base is None
        assert result.answer_verifier is None
        assert result.answer_extractor is None
        assert result.retrieval_top_k == 3

    def test_uses_injected_conversation(self, assistant, conversation):
        assert assistant.conversation is conversation

    def test_rejects_zero_retrieval_top_k(
        self, generator, prompt_builder
    ):
        with pytest.raises(ValueError):
            Assistant(
                generator=generator,
                prompt_builder=prompt_builder,
                retrieval_top_k=0,
            )

    def test_rejects_negative_retrieval_top_k(
        self, generator, prompt_builder
    ):
        with pytest.raises(ValueError):
            Assistant(
                generator=generator,
                prompt_builder=prompt_builder,
                retrieval_top_k=-1,
            )

    def test_requires_answer_extractor_when_knowledge_base_enabled(
        self, generator, prompt_builder
    ):
        knowledge_base = MagicMock()

        with pytest.raises(
            ValueError,
            match="AnswerExtractor is required",
        ):
            Assistant(
                generator=generator,
                prompt_builder=prompt_builder,
                knowledge_base=knowledge_base,
            )

    def test_creates_default_answer_verifier_when_knowledge_base_enabled(
        self, generator, prompt_builder
    ):
        knowledge_base = MagicMock()
        answer_extractor = MagicMock()

        result = Assistant(
            generator=generator,
            prompt_builder=prompt_builder,
            knowledge_base=knowledge_base,
            answer_extractor=answer_extractor,
        )

        assert result.answer_verifier is not None
        assert result.answer_extractor is answer_extractor


class TestAssistantNormalChat:

    def test_generates_response_without_knowledge_base(
        self, assistant, generator, prompt_builder, conversation
    ):
        conversation.get_history.return_value = [
            {"role": "user", "content": "Hello"}
        ]

        prompt_builder.build.return_value = "Generated prompt"
        generator.generate.return_value = "Hello! How can I help?"

        result = assistant.chat("How are you?")

        prompt_builder.build.assert_called_once_with(
            history=[{"role": "user", "content": "Hello"}],
            user_message="How are you?",
        )

        generator.generate.assert_called_once_with("Generated prompt")

        assert result == "Hello! How can I help?"

    def test_saves_user_and_assistant_messages(
        self, assistant, generator, prompt_builder, conversation
    ):
        prompt_builder.build.return_value = "Prompt"
        generator.generate.return_value = "Response"

        assistant.chat("Question")

        conversation.add_user_message.assert_called_once_with("Question")

        conversation.add_assistant_message.assert_called_once_with(
            "Response"
        )

    def test_returns_generated_response(
        self, assistant, generator, prompt_builder
    ):
        prompt_builder.build.return_value = "Prompt"
        generator.generate.return_value = "Expected response"

        result = assistant.chat("Question")

        assert result == "Expected response"

    @pytest.mark.parametrize(
        "message",
        ["", " ", "\n", "\t"],
    )
    def test_rejects_empty_user_message(self, assistant, message):
        with pytest.raises(
            ValueError,
            match="User message cannot be empty",
        ):
            assistant.chat(message)

    def test_does_not_generate_response_for_empty_message(
        self, assistant, generator
    ):
        with pytest.raises(ValueError):
            assistant.chat(" ")

        generator.generate.assert_not_called()


class TestAssistantConversation:

    def test_clear_conversation_delegates_to_conversation(
        self, assistant, conversation
    ):
        assistant.clear_conversation()

        conversation.clear.assert_called_once()


def create_rag_assistant():
    generator = MagicMock()
    prompt_builder = MagicMock()
    knowledge_base = MagicMock()
    context_validator = MagicMock()
    answer_extractor = MagicMock()
    answer_selector = MagicMock()
    def select_with_relevance(candidates, context):
        texts = [
            candidate["sentence"]
            if isinstance(candidate, dict)
            else candidate
            for candidate in candidates
        ]

        return SimpleNamespace(
            sentences=[{"sentence": sentence} for sentence in texts],
            texts=texts,
            top_relevance=1.0,
            aspects=[],
            uncovered_aspects=[],
            rejected=[],
        )

    answer_selector.select.side_effect = select_with_relevance

    knowledge_base.expand_to_parents.side_effect = (
        lambda documents: documents
    )

    assistant = Assistant(
        generator=generator,
        prompt_builder=prompt_builder,
        knowledge_base=knowledge_base,
        context_validator=context_validator,
        answer_extractor=answer_extractor,
        answer_selector=answer_selector,
        answer_verifier=MagicMock(),
        retrieval_top_k=5,
    )

    return (
        assistant,
        knowledge_base,
        context_validator,
        answer_extractor,
        generator,
    )


class TestAssistantRAGChat:

    def test_retrieves_documents_with_configured_top_k(self):
        (
            assistant,
            knowledge_base,
            context_validator,
            _,
            _,
        ) = create_rag_assistant()

        knowledge_base.search.return_value = []
        context_validator.is_relevant.return_value = False

        assistant.chat("What is Python?")

        knowledge_base.search.assert_called_once_with(
            query="What is Python?",
            top_k=5,
        )

    def test_returns_no_knowledge_message_when_context_is_irrelevant(self):
        (
            assistant,
            knowledge_base,
            context_validator,
            answer_extractor,
            generator,
        ) = create_rag_assistant()

        knowledge_base.search.return_value = [
            create_document("doc-1", "Unrelated information")
        ]

        context_validator.is_relevant.return_value = False

        result = assistant.chat("What is Python?")

        assert result == Assistant.NO_KNOWLEDGE_MESSAGE
        answer_extractor.extract_candidates.assert_not_called()
        generator.generate.assert_not_called()

    def test_returns_no_knowledge_message_when_formatted_context_is_empty(
        self
    ):
        (
            assistant,
            knowledge_base,
            context_validator,
            answer_extractor,
            _,
        ) = create_rag_assistant()

        knowledge_base.search.return_value = [
            create_document("doc-1", "   ")
        ]

        context_validator.is_relevant.return_value = True

        result = assistant.chat("What is Python?")

        assert result == Assistant.NO_KNOWLEDGE_MESSAGE
        answer_extractor.extract_candidates.assert_not_called()

    def test_returns_unsupported_message_when_extraction_is_empty(self):
        (
            assistant,
            knowledge_base,
            context_validator,
            answer_extractor,
            _,
        ) = create_rag_assistant()

        knowledge_base.search.return_value = [
            create_document(
                "doc-1",
                "Python is a programming language.",
                {"file_name": "python.txt"},
            )
        ]

        context_validator.is_relevant.return_value = True
        answer_extractor.extract_candidates.return_value = []

        result = assistant.chat("What is Python?")

        assert result == Assistant.UNSUPPORTED_ANSWER_MESSAGE
        answer_extractor.extract_candidates.assert_called_once()

    def test_returns_extracted_answer(self):
        (
            assistant,
            knowledge_base,
            context_validator,
            answer_extractor,
            generator,
        ) = create_rag_assistant()

        knowledge_base.search.return_value = [
            create_document(
                "doc-1",
                "Python is a programming language.",
                {"file_name": "python.txt"},
            )
        ]

        context_validator.is_relevant.return_value = True

        answer_extractor.extract_candidates.return_value = [
            "Python is a programming language.",
            "Python supports object-oriented programming.",
        ]

        result = assistant.chat("What is Python?")

        assert result == (
            "Python is a programming language. "
            "Python supports object-oriented programming."
        )

        generator.generate.assert_not_called()

    def test_passes_formatted_context_to_extractor(self):
        (
            assistant,
            knowledge_base,
            context_validator,
            answer_extractor,
            _,
        ) = create_rag_assistant()

        documents = [
            create_document(
                "doc-1",
                "Python is a programming language.",
                {"file_name": "python.txt"},
            )
        ]

        knowledge_base.search.return_value = documents
        context_validator.is_relevant.return_value = True

        answer_extractor.extract_candidates.return_value = [
            "Python is a language."
        ]

        assistant.chat("What is Python?")

        answer_extractor.extract_candidates.assert_called_once_with(
            question="What is Python?",
            context=(
                "[Document 1 | Source: python.txt]\n"
                "Python is a programming language."
            ),
        )


class TestFormatContext:

    def test_returns_none_for_empty_documents(self):
        assert Assistant._format_context([]) is None

    def test_returns_none_when_all_documents_are_empty(self):
        documents = [
            {"content": ""},
            {"content": "   "},
        ]

        assert Assistant._format_context(documents) is None

    def test_formats_file_name_and_chunk_index(self):
        documents = [
            {
                "content": "Python is a programming language.",
                "metadata": {
                    "file_name": "python.txt",
                    "chunk_index": 2,
                },
            }
        ]

        result = Assistant._format_context(documents)

        assert result == (
            "[Document 1 | Source: python.txt | Chunk: 2]\n"
            "Python is a programming language."
        )

    def test_uses_source_when_file_name_is_missing(self):
        documents = [
            {
                "content": "Some content.",
                "metadata": {"source": "manual.txt"},
            }
        ]

        result = Assistant._format_context(documents)

        assert result == (
            "[Document 1 | Source: manual.txt]\n"
            "Some content."
        )

    def test_uses_unknown_source_when_metadata_is_missing(self):
        documents = [
            {"content": "Some content."}
        ]

        result = Assistant._format_context(documents)

        assert result == (
            "[Document 1 | Source: Unknown source]\n"
            "Some content."
        )

    def test_handles_none_metadata(self):
        documents = [
            {
                "content": "Some content.",
                "metadata": None,
            }
        ]

        result = Assistant._format_context(documents)

        assert "Source: Unknown source" in result

    def test_skips_empty_documents_and_preserves_original_index(self):
        documents = [
            {"content": ""},
            {
                "content": "Second document.",
                "metadata": {"file_name": "second.txt"},
            },
        ]

        result = Assistant._format_context(documents)

        assert result == (
            "[Document 2 | Source: second.txt]\n"
            "Second document."
        )

    def test_formats_multiple_documents(self):
        documents = [
            {
                "content": "First document.",
                "metadata": {"file_name": "first.txt"},
            },
            {
                "content": "Second document.",
                "metadata": {"file_name": "second.txt"},
            },
        ]

        result = Assistant._format_context(documents)

        assert result == (
            "[Document 1 | Source: first.txt]\n"
            "First document.\n\n"
            "[Document 2 | Source: second.txt]\n"
            "Second document."
        )


class TestAssistantQueryDecomposition:

    def create_assistant(self):
        generator = MagicMock()
        prompt_builder = MagicMock()
        knowledge_base = MagicMock()
        context_validator = MagicMock()
        answer_extractor = MagicMock()
        answer_selector = MagicMock()
        query_decomposer = MagicMock()

        def select_with_relevance(candidates, context):
            texts = [
                candidate["sentence"]
                if isinstance(candidate, dict)
                else candidate
                for candidate in candidates
            ]

            return SimpleNamespace(
                sentences=[{"sentence": sentence} for sentence in texts],
                texts=texts,
                top_relevance=1.0,
                aspects=[],
                uncovered_aspects=[],
                rejected=[],
            )

        answer_selector.select.side_effect = select_with_relevance

        knowledge_base.expand_to_parents.side_effect = (
            lambda documents: documents
        )

        assistant = Assistant(
            generator=generator,
            prompt_builder=prompt_builder,
            knowledge_base=knowledge_base,
            context_validator=context_validator,
            answer_extractor=answer_extractor,
            answer_selector=answer_selector,
            answer_verifier=MagicMock(),
            query_decomposer=query_decomposer,
            retrieval_top_k=3,
        )

        return (
            assistant,
            knowledge_base,
            context_validator,
            answer_extractor,
            query_decomposer,
        )

    def test_simple_question_performs_one_retrieval(self):
        (
            assistant,
            knowledge_base,
            context_validator,
            answer_extractor,
            query_decomposer,
        ) = self.create_assistant()

        question = "What is Python?"

        document = create_document(
            "doc-1",
            "Python is a programming language.",
            {"file_name": "python.txt"},
        )

        query_decomposer.decompose.return_value = [question]
        knowledge_base.search.return_value = [document]

        context_validator.is_relevant.return_value = True

        answer_extractor.extract_candidates.return_value = [
            "Python is a programming language."
        ]

        result = assistant.chat(question)

        query_decomposer.decompose.assert_called_once_with(question)

        knowledge_base.search.assert_called_once_with(
            query=question,
            top_k=3,
        )

        assert result == "Python is a programming language."

    def test_complex_question_performs_multiple_retrievals(self):
        (
            assistant,
            knowledge_base,
            context_validator,
            answer_extractor,
            query_decomposer,
        ) = self.create_assistant()

        question = "What is RAG and how does it work?"

        query_decomposer.decompose.return_value = [
            "What is RAG?",
            "How does RAG work?",
        ]

        knowledge_base.search.side_effect = [
            [
                create_document(
                    "doc-1",
                    "RAG combines retrieval and generation.",
                )
            ],
            [
                create_document(
                    "doc-2",
                    "RAG retrieves relevant documents.",
                )
            ],
        ]

        context_validator.is_relevant.return_value = True

        answer_extractor.extract_candidates.return_value = [
            "RAG combines retrieval and generation.",
            "It retrieves relevant documents.",
        ]

        result = assistant.chat(question)

        assert knowledge_base.search.call_count == 2

        knowledge_base.search.assert_any_call(
            query="What is RAG?",
            top_k=3,
        )

        knowledge_base.search.assert_any_call(
            query="How does RAG work?",
            top_k=3,
        )

        answer_extractor.extract_candidates.assert_called_once()

        assert result == (
            "RAG combines retrieval and generation. "
            "It retrieves relevant documents."
        )

    def test_detailed_queries_are_passed_to_answer_extractor(self):
        (
            assistant,
            knowledge_base,
            context_validator,
            answer_extractor,
            _,
        ) = self.create_assistant()

        question = "What is RAG and how does it work?"

        class DetailedQueryDecomposer:
            def decompose_detailed(self, question):
                assert question == "What is RAG and how does it work?"
                return [
                    {
                        "aspect": "original",
                        "question": question,
                    },
                    {
                        "aspect": "definition",
                        "question": "What is RAG?",
                    },
                    {
                        "aspect": "workflow",
                        "question": "How does RAG work?",
                    },
                ]

            def decompose(self, question):
                raise AssertionError(
                    "Should use decompose_detailed() when available"
                )

        assistant.query_decomposer = DetailedQueryDecomposer()

        document = create_document(
            "doc-1",
            "RAG combines retrieval and generation.",
        )

        knowledge_base.search.return_value = [document]
        context_validator.is_relevant.return_value = True

        answer_extractor.extract_candidates.return_value = [
            "RAG combines retrieval and generation."
        ]

        assistant.chat(question)

        assert knowledge_base.search.call_count == 3

        answer_extractor.extract_candidates.assert_called_once()

        aspect_queries = (
            answer_extractor.extract_candidates.call_args.kwargs[
                "aspect_queries"
            ]
        )

        assert aspect_queries == [
            {
                "aspect": "definition",
                "question": "What is RAG?",
            },
            {
                "aspect": "workflow",
                "question": "How does RAG work?",
            },
        ]
    def test_retrieval_results_are_aggregated_by_rrf(self):
        (
            assistant,
            knowledge_base,
            context_validator,
            answer_extractor,
            query_decomposer,
        ) = self.create_assistant()

        question = "What is RAG and how does it work?"

        query_decomposer.decompose.return_value = [
            "What is RAG?",
            "How does RAG work?",
        ]

        document_a = create_document(
            "doc-A",
            "RAG combines retrieval and generation.",
        )

        document_b = create_document(
            "doc-B",
            "RAG retrieves relevant documents.",
        )

        document_c = create_document(
            "doc-C",
            "RAG generates answers using context.",
        )

        knowledge_base.search.side_effect = [
            [document_a, document_b],
            [
                create_document(
                    "doc-A",
                    " RAG combines retrieval and generation. ",
                ),
                document_c,
            ],
        ]

        context_validator.is_relevant.return_value = True

        answer_extractor.extract_candidates.return_value = [
            "RAG uses retrieved context."
        ]

        assistant.chat(question)

        context = answer_extractor.extract_candidates.call_args.kwargs["context"]

        # Same ID must appear only once.
        assert context.count(
            "RAG combines retrieval and generation."
        ) == 1

        assert "RAG retrieves relevant documents." in context
        assert "RAG generates answers using context." in context

        # The document retrieved by both queries should be first.
        assert context.index("RAG combines retrieval and generation.") < (
            context.index("RAG retrieves relevant documents.")
        )

    def test_returns_no_knowledge_when_all_retrievals_are_empty(self):
        (
            assistant,
            knowledge_base,
            context_validator,
            answer_extractor,
            query_decomposer,
        ) = self.create_assistant()

        question = "What is RAG and how does it work?"

        query_decomposer.decompose.return_value = [
            "What is RAG?",
            "How does RAG work?",
        ]

        knowledge_base.search.side_effect = [
            [],
            [],
        ]

        context_validator.is_relevant.return_value = False

        result = assistant.chat(question)

        assert result == Assistant.NO_KNOWLEDGE_MESSAGE
        assert knowledge_base.search.call_count == 2

        answer_extractor.extract_candidates.assert_not_called()

    def test_returns_unsupported_message_after_multiple_retrievals(self):
        (
            assistant,
            knowledge_base,
            context_validator,
            answer_extractor,
            query_decomposer,
        ) = self.create_assistant()

        question = "What is RAG and how does it work?"

        query_decomposer.decompose.return_value = [
            "What is RAG?",
            "How does RAG work?",
        ]

        knowledge_base.search.side_effect = [
            [
                create_document(
                    "doc-1",
                    "RAG is a retrieval augmented generation system.",
                )
            ],
            [
                create_document(
                    "doc-2",
                    "RAG retrieves relevant external information.",
                )
            ],
        ]

        context_validator.is_relevant.return_value = True
        answer_extractor.extract_candidates.return_value = []

        result = assistant.chat(question)

        assert result == Assistant.UNSUPPORTED_ANSWER_MESSAGE
        assert knowledge_base.search.call_count == 2

        answer_extractor.extract_candidates.assert_called_once()




from types import SimpleNamespace


def create_adaptive_rag_assistant():
    generator = MagicMock()
    prompt_builder = MagicMock()
    knowledge_base = MagicMock()
    answer_extractor = MagicMock()
    answer_selector = MagicMock()
    adaptive_retriever = MagicMock()
    query_plan_builder = MagicMock()
    evidence_context_builder = MagicMock()
    query_decomposer = MagicMock()

    def select_with_relevance(candidates, context):
        texts = [
            candidate["sentence"]
            if isinstance(candidate, dict)
            else candidate
            for candidate in candidates
        ]

        return SimpleNamespace(
            sentences=[{"sentence": sentence} for sentence in texts],
            texts=texts,
            top_relevance=1.0,
            aspects=[],
            uncovered_aspects=[],
            rejected=[],
        )

    answer_selector.select.side_effect = select_with_relevance

    query_plan = MagicMock()
    query_plan_builder.build.return_value = query_plan

    assistant = Assistant(
        generator=generator,
        prompt_builder=prompt_builder,
        knowledge_base=knowledge_base,
        answer_extractor=answer_extractor,
        answer_selector=answer_selector,
        answer_verifier=MagicMock(),
        adaptive_retriever=adaptive_retriever,
        query_plan_builder=query_plan_builder,
        evidence_context_builder=evidence_context_builder,
        query_decomposer=query_decomposer,
    )

    return SimpleNamespace(
        assistant=assistant,
        adaptive_retriever=adaptive_retriever,
        query_plan_builder=query_plan_builder,
        evidence_context_builder=evidence_context_builder,
        answer_extractor=answer_extractor,
        answer_selector=answer_selector,
        generator=generator,
        query_decomposer=query_decomposer,
    )


class TestAssistantAdaptiveRAG:

    def test_query_decomposition_and_context_are_used_for_adaptive_retrieval(self):
        components = create_adaptive_rag_assistant()
        components.query_decomposer.decompose_detailed.return_value = [
            {"aspect": "cause", "question": "What causes overfitting?"},
            {"aspect": "prevention", "question": "How can overfitting be prevented?"},
        ]

        evidence = [
            SimpleNamespace(
                evidence_type=SimpleNamespace(value="primary"),
                document=SimpleNamespace(
                    chunk=SimpleNamespace(
                        chunk_id="chunk-1",
                        content="Overfitting happens when a model memorizes training examples.",
                    )
                ),
            )
        ]
        components.adaptive_retriever.retrieve.return_value = SimpleNamespace(
            evidence=evidence,
            decision=SimpleNamespace(sufficient=True),
            iterations=1,
            stop_reason="THRESHOLD_REACHED",
        )
        components.evidence_context_builder.build.return_value = (
            "[Document 1 | Source: examples.txt | Type: PRIMARY]\n"
            "Overfitting happens when a model memorizes training examples."
        )
        components.answer_extractor.extract_candidates.return_value = [
            {"sentence": "Overfitting happens when a model memorizes training examples.", "score": 0.9}
        ]
        components.answer_selector.select.return_value = SimpleNamespace(
            sentences=[{"sentence": "Overfitting happens when a model memorizes training examples."}],
            texts=["Overfitting happens when a model memorizes training examples."],
            top_relevance=0.9,
            aspects=[],
            uncovered_aspects=[],
            rejected=[],
        )

        result = components.assistant.chat("What causes overfitting?")

        assert result == "Overfitting happens when a model memorizes training examples."
        components.query_decomposer.decompose_detailed.assert_called_once_with("What causes overfitting?")
        components.query_plan_builder.build.assert_called_once()
        build_kwargs = components.query_plan_builder.build.call_args.kwargs
        assert build_kwargs["original_question"] == "What causes overfitting?"
        assert [item["question"] for item in build_kwargs["decomposed_questions"]] == [
            "What causes overfitting?",
            "How can overfitting be prevented?",
        ]
        components.evidence_context_builder.build.assert_called_once_with(evidence)

    def test_fallback_context_extracts_real_evidence_text_without_object_serialization(self):
        assistant = Assistant(
            generator=MagicMock(),
            prompt_builder=MagicMock(),
            adaptive_retriever=MagicMock(),
            query_plan_builder=MagicMock(),
            evidence_context_builder=None,
            answer_extractor=MagicMock(),
            answer_selector=MagicMock(),
            answerability_checker=MagicMock(),
        )

        evidence = [
            SimpleNamespace(
                evidence_type=SimpleNamespace(value="primary"),
                document=SimpleNamespace(
                    chunk=SimpleNamespace(content="Actual evidence text from nested chunk.")
                ),
            )
        ]
        assistant.adaptive_retriever.retrieve.return_value = SimpleNamespace(
            evidence=evidence,
            decision=SimpleNamespace(sufficient=True),
            iterations=1,
            stop_reason="OK",
        )
        assistant.answer_extractor.extract_candidates.return_value = [
            {"sentence": "Actual evidence text from nested chunk.", "score": 0.8}
        ]
        assistant.answer_selector.select.return_value = SimpleNamespace(
            sentences=[{"sentence": "Actual evidence text from nested chunk."}],
            texts=["Actual evidence text from nested chunk."],
            top_relevance=0.8,
            aspects=[],
            uncovered_aspects=[],
            rejected=[],
        )
        assistant.answerability_checker.assess.return_value = SimpleNamespace(
            status="SUFFICIENT",
        )

        result = assistant.chat("What is relevant?")

        assert result == "Actual evidence text from nested chunk."
        assert "SimpleNamespace" not in assistant.last_context
        assert "Actual evidence text from nested chunk." in assistant.last_context
        assert "chunk" in assistant.last_context.lower()

    def test_returns_no_knowledge_when_no_evidence_is_found(self):
        components = create_adaptive_rag_assistant()

        components.adaptive_retriever.retrieve.return_value = (
            SimpleNamespace(
                evidence=[],
                decision=SimpleNamespace(sufficient=False),
                iterations=1,
                stop_reason="NO_NEW_EVIDENCE",
            )
        )

        result = components.assistant.chat("What is Python?")

        assert result == Assistant.NO_KNOWLEDGE_MESSAGE

        components.adaptive_retriever.retrieve.assert_called_once()
        components.answer_extractor.extract_candidates.assert_not_called()

    def test_returns_unsupported_when_evidence_is_insufficient(self):
        components = create_adaptive_rag_assistant()

        evidence = [MagicMock()]

        components.adaptive_retriever.retrieve.return_value = (
            SimpleNamespace(
                evidence=evidence,
                decision=SimpleNamespace(sufficient=False),
                iterations=2,
                stop_reason="MAX_ITERATIONS_REACHED",
            )
        )

        result = components.assistant.chat("What is Python?")

        assert result == Assistant.UNSUPPORTED_ANSWER_MESSAGE

        components.evidence_context_builder.build.assert_not_called()
        components.answer_extractor.extract_candidates.assert_not_called()

    def test_extracts_answer_when_evidence_is_sufficient(self):
        components = create_adaptive_rag_assistant()

        evidence = [MagicMock()]
        context = (
            "[Document 1 | Source: python.txt | Type: PRIMARY]\n"
            "Python is a programming language."
        )

        components.adaptive_retriever.retrieve.return_value = (
            SimpleNamespace(
                evidence=evidence,
                decision=SimpleNamespace(sufficient=True),
                iterations=1,
                stop_reason="THRESHOLD_REACHED",
            )
        )

        components.evidence_context_builder.build.return_value = context

        components.answer_extractor.extract_candidates.return_value = [
            "Python is a programming language."
        ]

        result = components.assistant.chat("What is Python?")

        assert result == "Python is a programming language."

        components.evidence_context_builder.build.assert_called_once_with(
            evidence
        )

        components.answer_extractor.extract_candidates.assert_called_once_with(
            question="What is Python?",
            context=context,
        )

        components.generator.generate.assert_not_called()

