from unittest.mock import MagicMock, patch

from cognexa.cli.assistant.commands import (
    create_vector_store,
    create_embedder,
    create_generator,
    create_prompt_builder,
)
from unittest.mock import MagicMock, patch

from cognexa.cli.assistant.commands import (
    create_vector_store,
    create_embedder,
    create_generator,
    create_prompt_builder,
    create_knowledge_base,
    create_ingestion_pipeline,
    create_assistant,
)

class TestBasicFactories:

    @patch("cognexa.cli.assistant.commands.VectorStore")
    def test_create_vector_store(self, mock_vector_store):
        result = create_vector_store()

        mock_vector_store.assert_called_once_with(
            persist_directory="data/vector_store",
            collection_name="cognexa_knowledge",
        )

        assert result is mock_vector_store.return_value

    @patch("cognexa.cli.assistant.commands.TextEmbedder")
    def test_create_embedder(self, mock_embedder):
        result = create_embedder()

        mock_embedder.assert_called_once_with()

        assert result is mock_embedder.return_value

    @patch("cognexa.cli.assistant.commands.TextGenerator")
    @patch("cognexa.cli.assistant.commands.TextTokenizer")
    @patch("cognexa.cli.assistant.commands.LLM")
    @patch("cognexa.cli.assistant.commands.LLMConfig")
    def test_create_generator(
        self,
        mock_config,
        mock_llm,
        mock_tokenizer,
        mock_generator,
    ):
        config = mock_config.return_value

        config.model_name = "test-model"
        config.max_new_tokens = 128
        config.temperature = 0.7
        config.top_k = 50
        config.top_p = 0.95
        config.repetition_penalty = 1.1
        config.do_sample = True

        result = create_generator()

        mock_llm.assert_called_once_with(
            model_name="test-model",
        )

        mock_tokenizer.assert_called_once_with(
            model_name="test-model",
        )

        mock_generator.assert_called_once_with(
            model=mock_llm.return_value,
            tokenizer=mock_tokenizer.return_value,
            max_new_tokens=128,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
            do_sample=True,
        )

        assert result is mock_generator.return_value

    @patch("cognexa.cli.assistant.commands.PromptBuilder")
    def test_create_prompt_builder(self, mock_prompt_builder):
        result = create_prompt_builder()

        mock_prompt_builder.assert_called_once_with(
            system_prompt="You are Cognexa, a helpful AI assistant."
        )

        assert result is mock_prompt_builder.return_value

class TestKnowledgeFactories:

    @patch("cognexa.cli.assistant.commands.KnowledgeBase")
    @patch("cognexa.cli.assistant.commands.BM25Retriever")
    @patch("cognexa.cli.assistant.commands.IngestionPipeline")
    @patch("cognexa.cli.assistant.commands.Retriever")
    @patch("cognexa.cli.assistant.commands.TextChunker")
    @patch("cognexa.cli.assistant.commands.DocumentLoader")
    @patch("cognexa.cli.assistant.commands.VectorStore")
    def test_create_knowledge_base_shares_dependencies(
        self,
        mock_vector_store,
        mock_loader,
        mock_chunker,
        mock_retriever,
        mock_pipeline,
        mock_bm25_retriever,
        mock_knowledge_base,
    ):
        embedder = MagicMock()

        result = create_knowledge_base(embedder=embedder)

        vector_store = mock_vector_store.return_value

        mock_vector_store.assert_called_once_with(
            persist_directory="data/vector_store",
            collection_name="cognexa_knowledge",
        )

        mock_retriever.assert_called_once_with(
            embedder=embedder,
            vector_store=vector_store,
        )

        mock_pipeline.assert_called_once()

        pipeline_kwargs = mock_pipeline.call_args.kwargs

        assert pipeline_kwargs["loader"] is mock_loader.return_value
        assert pipeline_kwargs["chunker"] is mock_chunker.return_value
        assert pipeline_kwargs["embedder"] is embedder
        assert pipeline_kwargs["vector_store"] is vector_store

        mock_bm25_retriever.assert_called_once_with()

        mock_knowledge_base.assert_called_once_with(
            ingestion_pipeline=mock_pipeline.return_value,
            retriever=mock_retriever.return_value,
            sparse_retriever=mock_bm25_retriever.return_value,
        )

        mock_knowledge_base.return_value.refresh_sparse_index.assert_called_once_with()

        assert result is mock_knowledge_base.return_value

    @patch("cognexa.cli.assistant.commands.KnowledgeBase")
    @patch("cognexa.cli.assistant.commands.BM25Retriever")
    @patch("cognexa.cli.assistant.commands.IngestionPipeline")
    @patch("cognexa.cli.assistant.commands.Retriever")
    @patch("cognexa.cli.assistant.commands.TextEmbedder")
    @patch("cognexa.cli.assistant.commands.TextChunker")
    @patch("cognexa.cli.assistant.commands.DocumentLoader")
    @patch("cognexa.cli.assistant.commands.VectorStore")
    def test_create_knowledge_base_creates_embedder_when_missing(
        self,
        mock_vector_store,
        mock_loader,
        mock_chunker,
        mock_embedder,
        mock_retriever,
        mock_pipeline,
        mock_bm25_retriever,
        mock_knowledge_base,
    ):
        create_knowledge_base()

        embedder = mock_embedder.return_value

        mock_embedder.assert_called_once_with()

        mock_retriever.assert_called_once_with(
            embedder=embedder,
            vector_store=mock_vector_store.return_value,
        )

        pipeline_kwargs = mock_pipeline.call_args.kwargs

        assert pipeline_kwargs["embedder"] is embedder

    @patch("cognexa.cli.assistant.commands.IngestionPipeline")
    @patch("cognexa.cli.assistant.commands.TextEmbedder")
    @patch("cognexa.cli.assistant.commands.TextChunker")
    @patch("cognexa.cli.assistant.commands.DocumentLoader")
    @patch("cognexa.cli.assistant.commands.VectorStore")
    def test_create_ingestion_pipeline(
        self,
        mock_vector_store,
        mock_loader,
        mock_chunker,
        mock_embedder,
        mock_pipeline,
    ):
        result = create_ingestion_pipeline()

        mock_embedder.assert_called_once_with()

        mock_vector_store.assert_called_once_with(
            persist_directory="data/vector_store",
            collection_name="cognexa_knowledge",
        )

        mock_pipeline.assert_called_once_with(
            loader=mock_loader.return_value,
            chunker=mock_chunker.return_value,
            embedder=mock_embedder.return_value,
            vector_store=mock_vector_store.return_value,
        )

        assert result is mock_pipeline.return_value


from unittest.mock import MagicMock, patch

from cognexa.cli.assistant.commands import create_assistant


class TestAssistantFactory:

    @patch("cognexa.cli.assistant.commands.Assistant")
    @patch("cognexa.cli.assistant.commands.AnswerSelector")
    @patch("cognexa.cli.assistant.commands.AnswerExtractor")
    @patch("cognexa.cli.assistant.commands.CrossEncoderReranker")
    @patch("cognexa.cli.assistant.commands.CrossEncoderCandidateReranker")
    @patch("cognexa.cli.assistant.commands.AnswerVerifier")
    @patch("cognexa.cli.assistant.commands.create_knowledge_base")
    @patch("cognexa.cli.assistant.commands.create_embedder")
    @patch("cognexa.cli.assistant.commands.create_prompt_builder")
    @patch("cognexa.cli.assistant.commands.create_generator")
    def test_create_assistant(
        self,
        mock_generator_factory,
        mock_prompt_builder_factory,
        mock_embedder_factory,
        mock_knowledge_base_factory,
        mock_answer_verifier,
        mock_candidate_reranker,
        mock_reranker,
        mock_answer_extractor,
        mock_answer_selector,
        mock_assistant,
    ):
        result = create_assistant()

        generator = mock_generator_factory.return_value
        prompt_builder = mock_prompt_builder_factory.return_value
        embedder = mock_embedder_factory.return_value
        knowledge_base = mock_knowledge_base_factory.return_value
        answer_verifier = mock_answer_verifier.return_value
        reranker = mock_reranker.return_value
        answer_extractor = mock_answer_extractor.return_value
        answer_selector = mock_answer_selector.return_value

        # Verify dependency factories
        mock_generator_factory.assert_called_once_with()
        mock_prompt_builder_factory.assert_called_once_with()
        mock_embedder_factory.assert_called_once_with()

        # Verify shared embedder
        mock_knowledge_base_factory.assert_called_once()
        assert mock_knowledge_base_factory.call_args.kwargs["embedder"] is embedder
        assert "vector_store" in mock_knowledge_base_factory.call_args.kwargs

        # Verify AnswerExtractor dependencies
        mock_answer_extractor.assert_called_once_with(
            reranker=reranker,
            min_candidates=2,
            max_candidates=24,
            score_gap_threshold=3.0,
        )

        # Verify AnswerSelector dependencies
        mock_answer_selector.assert_called_once_with(
            answer_verifier=answer_verifier,
        )

        # Verify Assistant dependencies
        mock_assistant.assert_called_once()
        assistant_kwargs = mock_assistant.call_args.kwargs

        assert assistant_kwargs["generator"] is generator
        assert assistant_kwargs["prompt_builder"] is prompt_builder
        assert assistant_kwargs["knowledge_base"] is knowledge_base
        assert assistant_kwargs["answer_verifier"] is answer_verifier
        assert assistant_kwargs["answer_extractor"] is answer_extractor
        assert assistant_kwargs["answer_selector"] is answer_selector
        assert assistant_kwargs["retrieval_top_k"] == 3
        assert "retrieval_reranker" not in assistant_kwargs
        assert "reranking_top_k" not in assistant_kwargs

        assert assistant_kwargs["adaptive_retriever"] is not None
        assert assistant_kwargs["query_plan_builder"] is not None
        assert assistant_kwargs["evidence_context_builder"] is not None

        assert result is mock_assistant.return_value









