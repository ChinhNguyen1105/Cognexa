from cognexa.assistant.assistant import Assistant
from cognexa.assistant.prompt import PromptBuilder
from cognexa.assistant.answer_extractor import AnswerExtractor
from cognexa.assistant.answer_selector import AnswerSelector
from cognexa.assistant.answerability_checker import AnswerabilityChecker
from cognexa.assistant.diagnostics import attach_diagnostics

from cognexa.llm.config import LLMConfig
from cognexa.llm.generator import TextGenerator
from cognexa.llm.model import LLM
from cognexa.llm.tokenizer import TextTokenizer

from cognexa.knowledge.knowledge_base import KnowledgeBase
from cognexa.knowledge.ingestion.loader import DocumentLoader
from cognexa.knowledge.ingestion.pipeline import IngestionPipeline
from cognexa.knowledge.processing.chunker import TextChunker
from cognexa.knowledge.embedding.embedder import TextEmbedder
from cognexa.knowledge.vector_store.store import VectorStore

from cognexa.knowledge.retrieval.candidates.dense_retriever import (
    Retriever,
    DenseRetriever,
)
from cognexa.knowledge.retrieval.candidates.sparse_retriever import (
    BM25Retriever,
)
from cognexa.knowledge.retrieval.ranking.reranker import (
    CrossEncoderReranker,
)
from cognexa.knowledge.retrieval.ranking.cross_encoder_reranker import (
    CrossEncoderCandidateReranker,
)
from cognexa.knowledge.retrieval.fusion.rrf_fusion import (
    ReciprocalRankFusion,
)
from cognexa.knowledge.retrieval.expansion.evidence_expander import (
    EvidenceExpander,
)
from cognexa.knowledge.retrieval.coverage.coverage_analyzer import (
    CoverageAnalyzer,
)
from cognexa.knowledge.retrieval.coverage.sufficiency_checker import (
    SufficiencyChecker,
)
from cognexa.knowledge.retrieval.adaptive_retriever import (
    AdaptiveRetriever,
)
from cognexa.knowledge.retrieval.query_plan_builder import (
    QueryPlanBuilder,
)
from cognexa.knowledge.retrieval.evidence_context_builder import (
    EvidenceContextBuilder,
)

from cognexa.verification import AnswerVerifier


VECTOR_STORE_PATH = "data/vector_store"
COLLECTION_NAME = "cognexa_knowledge"

# Refusal calibration for the answer flow. None disables the check.
# Read "top_relevance" from the [DEBUG ANSWERABILITY CHECKER] output for
# answerable and out-of-scope questions, then set a value between the two.
MIN_TOP_RELEVANCE = None
# Refuse when verified evidence covers less than this share of the aspects
# of a decomposed question (0.0 still refuses when no aspect is covered).
MIN_ASPECT_COVERAGE = 0.0


def create_vector_store() -> VectorStore:
    """
    Create the persistent vector store shared by
    document ingestion and knowledge retrieval.
    """
    return VectorStore(
        persist_directory=VECTOR_STORE_PATH,
        collection_name=COLLECTION_NAME,
    )


def create_embedder() -> TextEmbedder:
    """
    Create the embedding model used by Cognexa.
    """
    return TextEmbedder()


def create_generator() -> TextGenerator:
    """
    Create the LLM text generator.
    """
    config = LLMConfig()

    model = LLM(
        model_name=config.model_name,
    )

    tokenizer = TextTokenizer(
        model_name=config.model_name,
    )

    return TextGenerator(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        do_sample=config.do_sample,
    )


def create_prompt_builder() -> PromptBuilder:
    """
    Create the system prompt builder.
    """
    return PromptBuilder(
        system_prompt="You are Cognexa, a helpful AI assistant."
    )


def create_knowledge_base(
    embedder: TextEmbedder | None = None,
    *,
    vector_store: VectorStore | None = None,
    initialize_sparse_index: bool = True,
) -> KnowledgeBase:
    """
    Create the KnowledgeBase used by the AI Assistant.

    The vector store can be injected to ensure that
    ingestion and retrieval share the same instance.
    """
    embedder = embedder or create_embedder()
    vector_store = vector_store or create_vector_store()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    ingestion_pipeline = IngestionPipeline(
        loader=DocumentLoader(),
        chunker=TextChunker(),
        embedder=embedder,
        vector_store=vector_store,
    )

    sparse_retriever = BM25Retriever()

    knowledge_base = KnowledgeBase(
        ingestion_pipeline=ingestion_pipeline,
        retriever=retriever,
        sparse_retriever=sparse_retriever,
    )

    if initialize_sparse_index:
        indexed_count = knowledge_base.refresh_sparse_index()
        print(f"BM25 index initialized: {indexed_count} documents.")

    return knowledge_base


def create_ingestion_pipeline() -> IngestionPipeline:
    """
    Create the pipeline used to ingest documents.
    """
    return IngestionPipeline(
        loader=DocumentLoader(),
        chunker=TextChunker(),
        embedder=create_embedder(),
        vector_store=create_vector_store(),
    )


def create_assistant() -> Assistant:
    """
    Create the Adaptive RAG-enabled Cognexa Assistant.
    """
    generator = create_generator()
    prompt_builder = create_prompt_builder()

    # Shared infrastructure.
    embedder = create_embedder()
    vector_store = create_vector_store()

    knowledge_base = create_knowledge_base(
        embedder=embedder,
        vector_store=vector_store,
    )

    # Answer verification and extraction.
    answer_verifier = AnswerVerifier()

    # One CrossEncoder instance shared by both components.
    reranker = CrossEncoderReranker()

    answer_extractor = AnswerExtractor(
        reranker=reranker,
        min_candidates=2,
        max_candidates=24,
        score_gap_threshold=3.0,
    )

    answer_selector = AnswerSelector(
        answer_verifier=answer_verifier,
    )

    # Adaptive retrieval pipeline.
    dense_retriever = DenseRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    sparse_retriever = knowledge_base.sparse_retriever

    fusion = ReciprocalRankFusion()

    candidate_reranker = CrossEncoderCandidateReranker(
        scorer=reranker,
    )

    evidence_expander = EvidenceExpander(
        vector_store=vector_store,
    )

    coverage_analyzer = CoverageAnalyzer()

    sufficiency_checker = SufficiencyChecker(
        min_coverage=0.7,
    )

    adaptive_retriever = AdaptiveRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        fusion=fusion,
        reranker=candidate_reranker,
        expander=evidence_expander,
        coverage_analyzer=coverage_analyzer,
        sufficiency_checker=sufficiency_checker,
        top_k=5,
        max_iterations=3,
    )

    answerability_checker = AnswerabilityChecker(
        min_top_relevance=MIN_TOP_RELEVANCE,
        min_aspect_coverage=MIN_ASPECT_COVERAGE,
    )

    attach_diagnostics(
        adaptive_retriever=adaptive_retriever,
        answer_extractor=answer_extractor,
        answer_selector=answer_selector,
        answer_verifier=answer_verifier,
        answerability_checker=answerability_checker,
    )

    return Assistant(
        generator=generator,
        prompt_builder=prompt_builder,
        knowledge_base=knowledge_base,
        answer_verifier=answer_verifier,
        answer_extractor=answer_extractor,
        answer_selector=answer_selector,
        answerability_checker=answerability_checker,
        adaptive_retriever=adaptive_retriever,
        query_plan_builder=QueryPlanBuilder(),
        evidence_context_builder=EvidenceContextBuilder(),
        retrieval_top_k=3,
    )


def add_document(file_path: str):
    """
    Add a document to the knowledge base.
    """
    knowledge_base = create_knowledge_base(
        initialize_sparse_index=False,
    )

    result = knowledge_base.add_document(file_path)

    if result["chunk_count"] == 0:
        print("No text content found in the document.")
        return

    print("Document added successfully.")
    print(f"File: {result['source']}")
    print(f"Type: {result['file_type']}")
    print(f"Parent chunks created: {result['parent_count']}")
    print(f"Child chunks created: {result['child_count']}")


def chat():
    assistant = create_assistant()

    print("Cognexa Assistant")
    print("RAG mode: enabled")
    print("Retrieval mode: Adaptive RAG")
    print("Type 'exit' to quit.")
    print()

    while True:
        user_message = input("You: ")

        if user_message.lower() == "exit":
            break

        if not user_message.strip():
            continue

        response = assistant.chat(user_message)

        print(f"Cognexa: {response}")
        print()


