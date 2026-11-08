from cognexa.assistant.assistant import Assistant
from cognexa.assistant.prompt import PromptBuilder
from cognexa.assistant.answer_extractor import AnswerExtractor
from cognexa.assistant.answer_selector import AnswerSelector

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
from cognexa.knowledge.retrieval.retriever import Retriever
from cognexa.knowledge.validation.answer_verifier import AnswerVerifier
from cognexa.knowledge.reranking.reranker import (
    CrossEncoderReranker,
)

VECTOR_STORE_PATH = "data/vector_store"
COLLECTION_NAME = "cognexa_knowledge"


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
) -> KnowledgeBase:
    """
    Create the KnowledgeBase used by the AI Assistant.
    """
    embedder = embedder or create_embedder()
    vector_store = create_vector_store()

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

    return KnowledgeBase(
        ingestion_pipeline=ingestion_pipeline,
        retriever=retriever,
    )


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
    Create the RAG-enabled Cognexa Assistant.
    """
    generator = create_generator()
    prompt_builder = create_prompt_builder()

    embedder = create_embedder()

    knowledge_base = create_knowledge_base(
        embedder=embedder,
    )

    answer_verifier = AnswerVerifier()

    # Share one CrossEncoder instance between
    # retrieval reranking and answer extraction.
    reranker = CrossEncoderReranker()

    # Extractor: generate and rank sentence candidates.
    answer_extractor = AnswerExtractor(
        reranker=reranker,
        min_candidates=2,
        max_candidates=24,
        score_gap_threshold=3.0,
    )

    # Selector: verify evidence and select final sentences.
    answer_selector = AnswerSelector(
        answer_verifier=answer_verifier,
    )

    return Assistant(
        generator=generator,
        prompt_builder=prompt_builder,
        knowledge_base=knowledge_base,
        answer_verifier=answer_verifier,
        answer_extractor=answer_extractor,
        answer_selector=answer_selector,
        retrieval_reranker=reranker,
        retrieval_top_k=3,
        reranking_top_k=5,
    )


def add_document(file_path: str):
    """
    Add a document to the knowledge base.
    """
    pipeline = create_ingestion_pipeline()

    result = pipeline.ingest(file_path)

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
