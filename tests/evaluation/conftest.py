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
from cognexa.knowledge.vector_store.store import VectorStore

from cognexa.knowledge.retrieval.candidates.dense_retriever import (
    Retriever,
    DenseRetriever,
)
from cognexa.knowledge.retrieval.candidates.sparse_retriever import BM25Retriever
from cognexa.knowledge.retrieval.fusion.result_aggregator import ResultAggregator
from cognexa.knowledge.retrieval.fusion.rrf_fusion import ReciprocalRankFusion
from cognexa.knowledge.retrieval.ranking.reranker import CrossEncoderReranker
from cognexa.knowledge.retrieval.ranking.cross_encoder_reranker import (
    CrossEncoderCandidateReranker,
)
from cognexa.knowledge.retrieval.expansion.evidence_expander import EvidenceExpander
from cognexa.knowledge.retrieval.coverage.coverage_analyzer import CoverageAnalyzer
from cognexa.knowledge.retrieval.coverage.sufficiency_checker import SufficiencyChecker
from cognexa.knowledge.retrieval.adaptive_retriever import AdaptiveRetriever
from cognexa.knowledge.retrieval.query_plan_builder import QueryPlanBuilder

from cognexa.verification import AnswerVerifier, VerificationDecision
from cognexa.knowledge.validation.context_validator import ContextValidator


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALUATION_DOCUMENT = PROJECT_ROOT / "documents" / "evaluation" / "overfitting.txt"
EVALUATION_DATASET = Path(__file__).parent / "datasets" / "overfitting.json"


class DeterministicQuestionDecomposer:
    """Deterministic decomposer for reproducible evaluation."""

    def decompose(self, question: str) -> list[str]:
        if not isinstance(question, str):
            raise TypeError("question must be a string.")

        question = question.strip()

        if not question:
            raise ValueError("question cannot be empty.")

        return [question]


class DeterministicPromptBuilder:
    """Small prompt builder used by evaluation fixtures."""

    def build(self, history, user_message: str, context: str | None = None):
        if not isinstance(user_message, str):
            raise TypeError("user_message must be a string.")

        system_prompt = "You are Cognexa, a helpful AI assistant."
        if context and isinstance(context, str) and context.strip():
            system_prompt = (
                f"{system_prompt}\n\n"
                "Retrieved context:\n"
                f"{context}\n\n"
                "Use the retrieved context to answer the user and clearly state any uncertainty."
            )

        messages = [{"role": "system", "content": system_prompt}]
        for entry in history or []:
            messages.append({
                "role": entry.get("role", "user"),
                "content": entry.get("content", ""),
            })
        messages.append({"role": "user", "content": user_message})
        return messages


class DeterministicGenerator:
    """Stable generator for deterministic evaluation tests."""

    def generate(self, messages, *, max_new_tokens: int | None = None) -> str:
        prompt_text = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}"
            for item in messages or []
        )
        return (
            "Based on the retrieved context, I provide the most relevant answer available. "
            f"Prompt summary: {prompt_text[:180]}"
        )


@pytest.fixture(scope="session")
def evaluation_dataset():
    import json

    if not EVALUATION_DATASET.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {EVALUATION_DATASET}"
        )

    with EVALUATION_DATASET.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


@pytest.fixture(scope="session")
def evaluation_document():
    if not EVALUATION_DOCUMENT.exists():
        raise FileNotFoundError(
            f"Evaluation document not found: {EVALUATION_DOCUMENT}"
        )

    return EVALUATION_DOCUMENT


@pytest.fixture(scope="session")
def rag_environment(tmp_path_factory, evaluation_document):
    """Build an isolated RAG environment for evaluation."""

    temporary_directory = tmp_path_factory.mktemp("cognexa_rag_evaluation")

    vector_store = VectorStore(
        persist_directory=temporary_directory / "vector_store",
        collection_name="evaluation_knowledge",
    )

    embedder = TextEmbedder()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    sparse_retriever = BM25Retriever()

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
        sparse_retriever=sparse_retriever,
    )

    ingestion_result = knowledge_base.add_document(
        file_path=evaluation_document,
    )

    if ingestion_result["child_count"] == 0:
        raise RuntimeError("Evaluation document produced no child chunks.")

    if ingestion_result["parent_count"] == 0:
        raise RuntimeError("Evaluation document produced no parent chunks.")

    indexed_count = knowledge_base.refresh_sparse_index()

    if indexed_count == 0:
        raise RuntimeError("BM25 sparse index contains no documents.")

    decomposer = DeterministicQuestionDecomposer()
    result_aggregator = ResultAggregator()
    reranker = CrossEncoderReranker()

    context_validator = ContextValidator(max_distance=0.55)

    answer_verifier = AnswerVerifier(
        decision=VerificationDecision(
            entailment_threshold=0.75,
        ),
        min_overlap=0.15,
    )

    answer_extractor = AnswerExtractor(
        reranker=reranker,
        min_candidates=2,
        max_candidates=24,
        score_gap_threshold=3.0,
        min_score=None,
    )

    answer_selector = AnswerSelector(
        answer_verifier=answer_verifier,
        max_sentences=8,
        redundancy_threshold=0.85,
    )

    generator = DeterministicGenerator()
    prompt_builder = DeterministicPromptBuilder()

    candidate_reranker = CrossEncoderCandidateReranker(
        scorer=reranker,
    )

    adaptive_retriever = AdaptiveRetriever(
        dense_retriever=DenseRetriever(
            embedder=embedder,
            vector_store=vector_store,
        ),
        sparse_retriever=sparse_retriever,
        fusion=ReciprocalRankFusion(),
        reranker=candidate_reranker,
        expander=EvidenceExpander(
            vector_store=vector_store,
        ),
        coverage_analyzer=CoverageAnalyzer(),
        sufficiency_checker=SufficiencyChecker(min_coverage=0.7),
        top_k=5,
        max_iterations=3,
    )

    return SimpleNamespace(
        vector_store=vector_store,
        embedder=embedder,
        retriever=retriever,
        sparse_retriever=sparse_retriever,
        knowledge_base=knowledge_base,
        decomposer=decomposer,
        result_aggregator=result_aggregator,
        reranker=reranker,
        context_validator=context_validator,
        answer_verifier=answer_verifier,
        answer_extractor=answer_extractor,
        answer_selector=answer_selector,
        generator=generator,
        prompt_builder=prompt_builder,
        adaptive_retriever=adaptive_retriever,
        query_plan_builder=QueryPlanBuilder(),
        ingestion_result=ingestion_result,
        temporary_directory=temporary_directory,
    )


@pytest.fixture(scope="session")
def adaptive_rag_environment(rag_environment):
    """Build Adaptive RAG using the isolated evaluation components."""

    candidate_reranker = CrossEncoderCandidateReranker(
        scorer=rag_environment.reranker,
    )

    adaptive_retriever = AdaptiveRetriever(
        dense_retriever=DenseRetriever(
            embedder=rag_environment.embedder,
            vector_store=rag_environment.vector_store,
        ),
        sparse_retriever=rag_environment.sparse_retriever,
        fusion=ReciprocalRankFusion(),
        reranker=candidate_reranker,
        expander=EvidenceExpander(
            vector_store=rag_environment.vector_store,
        ),
        coverage_analyzer=CoverageAnalyzer(),
        sufficiency_checker=SufficiencyChecker(min_coverage=0.7),
        top_k=5,
        max_iterations=3,
    )

    return SimpleNamespace(
        **vars(rag_environment),
        adaptive_retriever=adaptive_retriever,
        query_plan_builder=QueryPlanBuilder(),
    )




