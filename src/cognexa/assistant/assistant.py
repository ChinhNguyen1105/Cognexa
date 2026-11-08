from cognexa.llm.generator import TextGenerator
from cognexa.knowledge.knowledge_base import KnowledgeBase
from cognexa.knowledge.query.question_decomposer import QuestionDecomposer
from cognexa.knowledge.retrieval.result_aggregator import ResultAggregator
from cognexa.knowledge.reranking.reranker import CrossEncoderReranker
from cognexa.knowledge.validation.context_validator import ContextValidator
from cognexa.knowledge.validation.answer_verifier import AnswerVerifier

from .conversation import Conversation
from .prompt import PromptBuilder
from .answer_extractor import AnswerExtractor
from .answer_selector import AnswerSelector
from .answerability_checker import AnswerabilityChecker


class Assistant:
    NO_KNOWLEDGE_MESSAGE = (
        "I could not find relevant information "
        "in the knowledge base to answer your question."
    )

    UNSUPPORTED_ANSWER_MESSAGE = (
        "I could not find sufficient verified information "
        "to answer your question."
    )

    def __init__(
        self,
        generator: TextGenerator,
        prompt_builder: PromptBuilder,
        conversation: Conversation | None = None,
        knowledge_base: KnowledgeBase | None = None,
        context_validator: ContextValidator | None = None,
        answer_verifier: AnswerVerifier | None = None,
        answer_extractor: AnswerExtractor | None = None,
        answer_selector: AnswerSelector | None = None,
        answerability_checker: AnswerabilityChecker | None = None,
        retrieval_top_k: int = 3,
        query_decomposer: QuestionDecomposer | None = None,
        result_aggregator: ResultAggregator | None = None,
        retrieval_reranker: CrossEncoderReranker | None = None,
        reranking_top_k: int = 5,
    ):
        if retrieval_top_k <= 0:
            raise ValueError(
                "retrieval_top_k must be greater than 0."
            )

        if reranking_top_k <= 0:
            raise ValueError(
                "reranking_top_k must be greater than 0."
            )

        self.generator = generator
        self.prompt_builder = prompt_builder
        self.conversation = conversation or Conversation()
        self.knowledge_base = knowledge_base

        self.context_validator = (
            context_validator or ContextValidator()
        )

        self.answer_verifier = answer_verifier
        self.answer_extractor = answer_extractor
        self.answer_selector = answer_selector

        self.answerability_checker = (
            answerability_checker or AnswerabilityChecker()
        )

        self.query_decomposer = query_decomposer

        self.result_aggregator = (
            result_aggregator or ResultAggregator()
        )

        self.retrieval_reranker = retrieval_reranker
        self.retrieval_top_k = retrieval_top_k
        self.reranking_top_k = reranking_top_k

        if self.knowledge_base is not None:
            if self.answer_verifier is None:
                self.answer_verifier = AnswerVerifier()

            if self.answer_extractor is None:
                raise ValueError(
                    "AnswerExtractor is required when "
                    "KnowledgeBase is enabled."
                )

            if self.answer_selector is None:
                self.answer_selector = AnswerSelector(
                    answer_verifier=self.answer_verifier,
                )

            if self.query_decomposer is None:
                self.query_decomposer = QuestionDecomposer(
                    generator=self.generator,
                )

    def chat(self, user_message: str) -> str:
        if not isinstance(user_message, str):
            raise TypeError(
                "User message must be a string."
            )

        if not user_message.strip():
            raise ValueError(
                "User message cannot be empty."
            )

        history = self.conversation.get_history()

        if self.knowledge_base is None:
            prompt = self.prompt_builder.build(
                history=history,
                user_message=user_message,
            )

            response = self.generator.generate(prompt)

            return self._save_response(
                user_message,
                response,
            )

        print("\n[DEBUG ASSISTANT] Starting retrieval")

        retrieved_documents = self._retrieve_documents(
            user_message
        )

        print(
            f"[DEBUG ASSISTANT] Retrieved child chunks: "
            f"{len(retrieved_documents)}"
        )

        # Step 1: Validate retrieved child chunks.
        is_relevant = self.context_validator.is_relevant(
            retrieved_documents
        )

        print(
            f"\n[DEBUG CONTEXT VALIDATOR] "
            f"is_relevant = {is_relevant}"
        )

        if not is_relevant:
            return self._save_response(
                user_message,
                self.NO_KNOWLEDGE_MESSAGE,
            )

        # Step 2: Expand selected children into parent contexts.
        expanded_documents = (
            self.knowledge_base.expand_to_parents(
                retrieved_documents
            )
        )

        print(
            f"\n[DEBUG PARENT EXPANSION] "
            f"Child chunks: {len(retrieved_documents)} | "
            f"Expanded documents: {len(expanded_documents)}"
        )

        # Step 3: Build context from expanded documents.
        context = self._format_context(
            expanded_documents
        )

        if context is None:
            return self._save_response(
                user_message,
                self.NO_KNOWLEDGE_MESSAGE,
            )

        # Step 4: Extract and rank answer candidates.
        print(
            "\n[DEBUG ASSISTANT] Calling AnswerExtractor"
        )

        if isinstance(self.answer_extractor, AnswerExtractor):
            candidates = self.answer_extractor.extract_candidates(
                question=user_message,
                context=context,
                aspect_queries=getattr(self, "_last_aspect_queries", None),
            )
        else:
            candidates = self.answer_extractor.extract_candidates(
                question=user_message,
                context=context,
            )

        print(
            f"[DEBUG ASSISTANT] Candidate sentences: "
            f"{len(candidates)}"
        )

        if not candidates:
            return self._save_response(
                user_message,
                self.UNSUPPORTED_ANSWER_MESSAGE,
            )

        # Step 5: Select relevant, supported and non-redundant sentences.
        print(
            "\n[DEBUG ASSISTANT] Calling AnswerSelector"
        )

        selection_result = None
        if isinstance(self.answer_selector, AnswerSelector):
            selection_result = self.answer_selector.select_detailed(
                candidates=candidates,
                context=context,
                question=user_message,
            )
            selected_sentences = selection_result.texts
        else:
            selected_sentences = self.answer_selector.select(
                candidates=candidates,
                context=context,
            )

        print(
            f"[DEBUG ASSISTANT] Selected sentences: "
            f"{len(selected_sentences)}"
        )

        # Step 6: Check whether a verified answer is available.
        if selection_result is not None:
            assessment = self.answerability_checker.assess(
                question=user_message,
                selection=selection_result,
            )
            is_answerable = assessment.answerable
        else:
            is_answerable = self.answerability_checker.is_answerable(
                selected_sentences
            )

        print(
            f"[DEBUG ANSWERABILITY CHECKER] "
            f"is_answerable = {is_answerable}"
        )

        if not is_answerable:
            return self._save_response(
                user_message,
                self.UNSUPPORTED_ANSWER_MESSAGE,
            )

        response = " ".join(selected_sentences)

        return self._save_response(
            user_message,
            response,
        )

    def _retrieve_documents(
        self,
        user_message: str,
    ) -> list[dict]:
        """
        Retrieve child chunks using multiple queries,
        aggregate them using RRF, then rerank with CrossEncoder.
        """

        # Preserve aspect metadata for answer extraction while keeping
        # retrieval queries as plain strings for backward compatibility.
        if isinstance(self.query_decomposer, QuestionDecomposer):
            detailed_queries = self.query_decomposer.decompose_detailed(
                user_message
            )
        else:
            # Preserve compatibility with mocks and custom decomposers
            # that implement only the original decompose() interface.
            detailed_queries = [
                {
                    "aspect": "original",
                    "question": query,
                }
                for query in self.query_decomposer.decompose(user_message)
            ]

        queries = [
            item["question"]
            for item in detailed_queries
        ]

        self._last_aspect_queries = [
            item
            for item in detailed_queries
            if item["aspect"] != "original"
        ]

        retrieval_results = []

        for query in queries:
            documents = self.knowledge_base.search(
                query=query,
                top_k=self.retrieval_top_k,
            )

            retrieval_results.append(documents)

        # Step 1: Aggregate retrieval results using RRF.
        aggregated_documents = (
            self.result_aggregator.aggregate(
                retrieval_results=retrieval_results,
            )
        )

        print(
            f"\n[DEBUG RRF] Aggregated child chunks: "
            f"{len(aggregated_documents)}"
        )

        if not aggregated_documents:
            return []

        # Step 2: Rerank child chunks using CrossEncoder.
        if self.retrieval_reranker is None:
            return aggregated_documents[
                :self.reranking_top_k
            ]

        candidates = [
            document["content"]
            for document in aggregated_documents
        ]

        ranked_candidates = (
            self.retrieval_reranker.rerank(
                query=user_message,
                candidates=candidates,
            )
        )

        reranked_documents = []

        print("\n[DEBUG CHILD RERANKING]")

        for candidate in ranked_candidates:
            document = dict(
                aggregated_documents[candidate["index"]]
            )

            document["rerank_score"] = candidate["score"]

            reranked_documents.append(document)

            print(
                f"Score: {candidate['score']:.4f} | "
                f"Distance: {document.get('distance')} | "
                f"Content: {document.get('content', '')[:150]}"
            )

        # Step 3: Select top-ranked child chunks.
        return reranked_documents[
            :self.reranking_top_k
        ]

    def _save_response(
        self,
        user_message: str,
        response: str,
    ) -> str:
        self.conversation.add_user_message(
            user_message
        )

        self.conversation.add_assistant_message(
            response
        )

        return response

    @staticmethod
    def _format_context(
        documents: list[dict],
    ) -> str | None:
        if not documents:
            return None

        context_parts = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            content = document.get("content", "")

            if (
                not isinstance(content, str)
                or not content.strip()
            ):
                continue

            metadata = document.get("metadata") or {}

            source = (
                metadata.get("file_name")
                or metadata.get("source")
                or "Unknown source"
            )

            parent_index = metadata.get("parent_index")
            chunk_index = metadata.get("chunk_index")

            source_info = f"Source: {source}"

            if parent_index is not None:
                source_info += (
                    f" | Parent: {parent_index}"
                )

            if chunk_index is not None:
                source_info += (
                    f" | Chunk: {chunk_index}"
                )

            context_parts.append(
                f"[Document {index} | {source_info}]\n"
                f"{content}"
            )

        if not context_parts:
            return None

        return "\n\n".join(context_parts)

    def clear_conversation(self):
        self.conversation.clear()
