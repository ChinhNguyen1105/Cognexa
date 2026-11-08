from pathlib import Path
from types import SimpleNamespace

import pytest

from cognexa.assistant.answer_extractor import AnswerExtractor
from cognexa.assistant.answer_selector import AnswerSelector
from cognexa.knowledge.embedding.embedder import TextEmbedder
from cognexa.knowledge.ingestion.loader import DocumentLoader
from cognexa.knowledge.ingestion.pipeline import IngestionPipeline
from cognexa.knowledge.knowledge_base import KnowledgeBase
from cognexa.knowledge.processing.chunker import TextChunker
from cognexa.knowledge.reranking.reranker import CrossEncoderReranker
from cognexa.knowledge.retrieval.result_aggregator import ResultAggregator
from cognexa.knowledge.retrieval.retriever import Retriever
from cognexa.knowledge.validation.answer_verifier import AnswerVerifier
from cognexa.knowledge.validation.context_validator import ContextValidator
from cognexa.knowledge.vector_store.store import VectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALUATION_DOCUMENT = (
    PROJECT_ROOT
    / "documents"
    / "evaluation"
    / "overfitting.txt"
)

EVALUATION_DATASET = (
    Path(__file__).parent
    / "datasets"
    / "overfitting.json"
)


class DeterministicQuestionDecomposer:
    """
    Deterministic decomposer for reproducible evaluation.

    Keeps the original question unchanged so that evaluation
    focuses on retrieval and answer quality rather than
    LLM-generated query variations.
    """

    def decompose(self, question: str) -> list[str]:
        if not isinstance(question, str):
            raise TypeError("question must be a string.")

        question = question.strip()

        if not question:
            raise ValueError("question cannot be empty.")

        return [question]


@pytest.fixture(scope="session")
def evaluation_dataset():
    """
    Load the evaluation question dataset.
    """

    import json

    if not EVALUATION_DATASET.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {EVALUATION_DATASET}"
        )

    with EVALUATION_DATASET.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        return json.load(file)


@pytest.fixture(scope="session")
def evaluation_document():
    """
    Return the source document used for evaluation.
    """

    if not EVALUATION_DOCUMENT.exists():
        raise FileNotFoundError(
            f"Evaluation document not found: {EVALUATION_DOCUMENT}"
        )

    return EVALUATION_DOCUMENT


@pytest.fixture(scope="session")
def rag_environment(
    tmp_path_factory,
    evaluation_document,
):
    """
    Build an isolated RAG environment using real components.

    The ChromaDB database is created in pytest's temporary
    directory. The project's production vector store remains
    untouched.
    """

    temporary_directory = tmp_path_factory.mktemp(
        "cognexa_rag_evaluation"
    )

    vector_store = VectorStore(
        persist_directory=temporary_directory / "vector_store",
        collection_name="evaluation_knowledge",
    )

    embedder = TextEmbedder()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    ingestion_pipeline = IngestionPipeline(
        loader=DocumentLoader(),
        chunker=TextChunker(
            chunk_size=500,
            chunk_overlap=50,
        ),
        embedder=embedder,
        vector_store=vector_store,
        parent_chunk_size=1500,
        parent_chunk_overlap=150,
    )

    knowledge_base = KnowledgeBase(
        ingestion_pipeline=ingestion_pipeline,
        retriever=retriever,
    )

    ingestion_result = knowledge_base.add_document(
        file_path=evaluation_document,
    )

    if ingestion_result["child_count"] == 0:
        raise RuntimeError(
            "Evaluation document produced no child chunks."
        )

    if ingestion_result["parent_count"] == 0:
        raise RuntimeError(
            "Evaluation document produced no parent chunks."
        )

    decomposer = DeterministicQuestionDecomposer()

    result_aggregator = ResultAggregator()

    reranker = CrossEncoderReranker()

    context_validator = ContextValidator(
        max_distance=0.55,
    )

    answer_verifier = AnswerVerifier(
        min_entailment_score=0.70,
        device="cpu",
    )

    answer_extractor = AnswerExtractor(
        reranker=reranker,
        min_candidates=2,
        max_candidates=24,
        score_gap_threshold=3.0,
    )

    answer_selector = AnswerSelector(
        answer_verifier=answer_verifier,
        max_sentences=8,
        redundancy_threshold=0.85,
    )

    return SimpleNamespace(
        vector_store=vector_store,
        embedder=embedder,
        retriever=retriever,
        knowledge_base=knowledge_base,
        decomposer=decomposer,
        result_aggregator=result_aggregator,
        reranker=reranker,
        context_validator=context_validator,
        answer_verifier=answer_verifier,
        answer_extractor=answer_extractor,
        answer_selector=answer_selector,
        ingestion_result=ingestion_result,
        temporary_directory=temporary_directory,
    )

