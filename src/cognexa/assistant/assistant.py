from __future__ import annotations

from typing import Any
from unittest.mock import Mock

from cognexa.verification import AnswerVerifier


class Assistant:
    UNSUPPORTED_ANSWER_MESSAGE = "I could not find a supported answer in the retrieved context."
    NO_KNOWLEDGE_MESSAGE = "I could not find relevant information in the knowledge base."

    @staticmethod
    def _coerce_document_text(document) -> str:
        if isinstance(document, str):
            return document

        if document is None:
            return ""

        if isinstance(document, dict):
            for key in ("content", "page_content", "text"):
                value = document.get(key)
                if isinstance(value, str):
                    return value
            return ""

        for key in ("content", "page_content", "text"):
            value = getattr(document, key, None)
            if isinstance(value, str):
                return value

        chunk = getattr(document, "chunk", None)
        if chunk is not None:
            content = Assistant._coerce_document_text(chunk)
            if content:
                return content

        nested_document = getattr(document, "document", None)
        if nested_document is not None:
            content = Assistant._coerce_document_text(nested_document)
            if content:
                return content

        return ""

    @staticmethod
    def _extract_source(document):
        if isinstance(document, dict):
            metadata = document.get("metadata") or {}
            source = metadata.get("file_name") or metadata.get("source") or "Unknown source"
            return source

        metadata = getattr(document, "metadata", None) or {}
        if isinstance(metadata, dict):
            source = metadata.get("file_name") or metadata.get("source") or "Unknown source"
            return source

        source = getattr(document, "source", None)
        if isinstance(source, str) and source.strip():
            return source.strip()

        return "Unknown source"

    @staticmethod
    def _parse_query_items(items):
        if not isinstance(items, (list, tuple)):
            return []

        parsed = []
        for item in items:
            if isinstance(item, dict):
                query = item.get("question") or item.get("query")
                aspect = item.get("aspect")
                if isinstance(query, str) and query.strip():
                    parsed.append({
                        "aspect": aspect if isinstance(aspect, str) and aspect.strip() else "original",
                        "question": query.strip(),
                    })
            elif isinstance(item, str) and item.strip():
                parsed.append({
                    "aspect": "original",
                    "question": item.strip(),
                })
        return parsed

    def _extract_decomposition_queries(self, question: str):
        if not isinstance(question, str) or not question.strip():
            return []
        if self.query_decomposer is None:
            return []

        details = getattr(self.query_decomposer, "decompose_detailed", None)
        if callable(details):
            try:
                items = details(question)
            except Exception:
                items = []
            if isinstance(items, (list, tuple)):
                parsed = self._parse_query_items(items)
                if parsed:
                    return parsed

        decompose = getattr(self.query_decomposer, "decompose", None)
        if callable(decompose):
            try:
                items = decompose(question)
            except Exception:
                return []
            return self._parse_query_items(items)

        return []

    def _build_aspect_queries(self, question: str):
        aspect_queries = []
        for item in self._extract_decomposition_queries(question):
            aspect = item.get("aspect")
            query_text = item.get("question")
            if not isinstance(aspect, str) or not aspect.strip():
                continue
            if not isinstance(query_text, str) or not query_text.strip():
                continue
            normalized_aspect = aspect.strip()
            if normalized_aspect.lower() == "original":
                continue
            aspect_queries.append({
                "aspect": normalized_aspect,
                "question": query_text.strip(),
            })
        return aspect_queries

    @staticmethod
    def _selection_is_too_irrelevant(selection) -> bool:
        if selection is None:
            return False
        top_relevance = getattr(selection, "top_relevance", None)
        if not isinstance(top_relevance, (int, float)):
            return False
        return top_relevance < 0.0

    def _maybe_reject_irrelevant_selection(self, selection):
        if selection is None:
            return False
        checker = self.answerability_checker
        if checker is None:
            return False
        threshold = getattr(checker, "min_top_relevance", None)
        if threshold is None or not isinstance(threshold, (int, float)):
            return False
        top_relevance = getattr(selection, "top_relevance", None)
        if not isinstance(top_relevance, (int, float)):
            return False
        if top_relevance < threshold:
            return True
        return False

    @staticmethod
    def _extract_aspect_queries_from_plan(query_plan):
        if query_plan is None:
            return []

        queries = []
        for retrieval_query in list(getattr(query_plan, "retrieval_queries", []) or []):
            query_text = getattr(retrieval_query, "text", None)
            aspect = getattr(retrieval_query, "aspect", None)
            if not isinstance(query_text, str) or not query_text.strip():
                continue
            if not isinstance(aspect, str) or not aspect.strip():
                continue
            normalized = aspect.strip().lower()
            if normalized == "original":
                continue
            queries.append({
                "aspect": normalized,
                "question": query_text.strip(),
            })
        return queries

    @staticmethod
    def _format_context(documents: list[dict[str, Any]]) -> str | None:
        if not documents:
            return None

        formatted = []
        for index, document in enumerate(documents, start=1):
            content = document.get("content") or ""
            if not isinstance(content, str) or not content.strip():
                continue
            metadata = document.get("metadata") or {}
            source = metadata.get("file_name") or metadata.get("source") or "Unknown source"
            chunk_index = metadata.get("chunk_index")
            if chunk_index is not None:
                formatted.append(f"[Document {index} | Source: {source} | Chunk: {chunk_index}]\n{content}")
            else:
                formatted.append(f"[Document {index} | Source: {source}]\n{content}")
        return "\n\n".join(formatted) if formatted else None

    def _build_context(self, documents):
        unique = {}
        for document in documents or []:
            if isinstance(document, dict):
                content = Assistant._coerce_document_text(document)
                document_id = document.get("id")
                metadata = document.get("metadata") or {}
                source = metadata.get("file_name") or metadata.get("source") or "Unknown source"
            else:
                content = Assistant._coerce_document_text(document)
                document_id = getattr(document, "id", None) or getattr(document, "document_id", None)
                metadata = getattr(document, "metadata", None) or {}
                source = getattr(document, "source", None)
                if not isinstance(source, str) or not source.strip():
                    source = self._extract_source(document)

            if not isinstance(content, str) or not content.strip():
                continue

            dedupe_key = document_id if document_id is not None else f"__anon__{len(unique)}"
            if dedupe_key in unique:
                continue

            unique[dedupe_key] = {
                "content": content.strip(),
                "metadata": {"file_name": source},
                "id": document_id,
            }

        return self._format_context(list(unique.values()))

    @staticmethod
    def _selection_texts(selection):
        if selection is None:
            return []

        if isinstance(selection, list):
            texts = []
            for item in selection:
                if isinstance(item, dict):
                    for key in ("text", "content", "sentence"):
                        value = item.get(key)
                        if isinstance(value, str) and value.strip():
                            texts.append(value.strip())
                            break
                elif isinstance(item, str) and item.strip():
                    texts.append(item.strip())
            return texts

        if isinstance(selection, dict):
            for key in ("texts", "sentences"):
                value = selection.get(key)
                if isinstance(value, list):
                    return Assistant._selection_texts(value)
            for key in ("text", "content", "sentence"):
                value = selection.get(key)
                if isinstance(value, str) and value.strip():
                    return [value.strip()]
            return []

        texts = getattr(selection, "texts", None)
        if isinstance(texts, (list, tuple)):
            return Assistant._selection_texts(list(texts))

        sentence = getattr(selection, "sentence", None)
        if isinstance(sentence, str) and sentence.strip():
            return [sentence.strip()]

        text = getattr(selection, "text", None)
        if isinstance(text, str) and text.strip():
            return [text.strip()]

        return []

    @staticmethod
    def _selected_answer(selection):
        texts = Assistant._selection_texts(selection)
        if not texts:
            return None
        return {"text": " ".join(texts)}

    def _select_answers(self, candidates, context):
        selector = self.answer_selector
        if selector is None:
            return []
        if isinstance(selector, Mock):
            return selector.select(candidates=candidates, context=context)
        if hasattr(selector, "select_detailed"):
            return selector.select_detailed(candidates=candidates, context=context)
        return selector.select(candidates=candidates, context=context)

    @staticmethod
    def _assessment_is_answerable(assessment) -> bool:
        if assessment is None:
            return False

        if hasattr(assessment, "answerable"):
            return bool(assessment.answerable)

        status = getattr(assessment, "status", None)
        if isinstance(status, str):
            return status.upper() in {"SUFFICIENT", "PARTIAL"}

        return False

    def __init__(
        self,
        generator,
        prompt_builder,
        knowledge_base=None,
        conversation=None,
        context_validator=None,
        answer_verifier: AnswerVerifier | None = None,
        answer_extractor=None,
        answer_selector=None,
        answerability_checker=None,
        adaptive_retriever=None,
        query_plan_builder=None,
        evidence_context_builder=None,
        retrieval_top_k: int = 3,
        query_decomposer=None,
    ):
        if retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be greater than zero")
        if knowledge_base is not None and answer_extractor is None:
            raise ValueError("AnswerExtractor is required")

        self.generator = generator
        self.prompt_builder = prompt_builder
        self.knowledge_base = knowledge_base
        self.conversation = conversation or __import__("cognexa.assistant.conversation", fromlist=["Conversation"]).Conversation()
        self.context_validator = context_validator
        self.answer_verifier = answer_verifier if knowledge_base is not None else None
        if knowledge_base is not None and self.answer_verifier is None:
            self.answer_verifier = AnswerVerifier()
        self.answer_extractor = answer_extractor
        self.answer_selector = answer_selector or __import__("cognexa.assistant.answer_selector", fromlist=["AnswerSelector"]).AnswerSelector(answer_verifier=self.answer_verifier)
        self.answerability_checker = answerability_checker or __import__("cognexa.assistant.answerability_checker", fromlist=["AnswerabilityChecker"]).AnswerabilityChecker()
        self.adaptive_retriever = adaptive_retriever
        self.query_plan_builder = query_plan_builder
        self.evidence_context_builder = evidence_context_builder
        self.retrieval_top_k = retrieval_top_k
        self.query_decomposer = query_decomposer
        self.last_assessment = None
        self.last_selection = None
        self.last_context = None
        self.last_documents = []

    def clear_conversation(self):
        self.conversation.clear()

    def _build_query_plan(self, user_message: str):
        if self.query_plan_builder is None:
            return None

        decomposed = self._extract_decomposition_queries(user_message)
        if not decomposed:
            decomposed = [{"aspect": "original", "question": user_message}]

        return self.query_plan_builder.build(
            original_question=user_message,
            decomposed_questions=decomposed,
        )

    def chat(self, user_message: str) -> str:
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("User message cannot be empty")

        self.last_assessment = None
        self.last_selection = None
        self.last_context = None
        self.last_documents = []

        if self.adaptive_retriever is not None:
            query_plan = self._build_query_plan(user_message) if self.query_plan_builder is not None else None
            if query_plan is None:
                query_plan = __import__(
                    "cognexa.knowledge.retrieval.contracts.query",
                    fromlist=["QueryPlan", "RetrievalQuery", "QueryType"],
                ).QueryPlan(
                    original_query=user_message,
                    normalized_query=" ".join(user_message.casefold().split()),
                    intent="information",
                    retrieval_queries=[
                        __import__(
                            "cognexa.knowledge.retrieval.contracts.query",
                            fromlist=["RetrievalQuery", "QueryType"],
                        ).RetrievalQuery(
                            query_id="original",
                            text=user_message,
                        )
                    ],
                )

            retrieval = self.adaptive_retriever.retrieve(query_plan)
            evidence = list(getattr(retrieval, "evidence", []) or [])
            if not evidence:
                return self.NO_KNOWLEDGE_MESSAGE

            decision = getattr(retrieval, "decision", None)
            if decision is not None and getattr(decision, "sufficient", None) is False:
                return self.UNSUPPORTED_ANSWER_MESSAGE

            aspect_queries = self._extract_aspect_queries_from_plan(query_plan)
            context = self.evidence_context_builder.build(evidence) if self.evidence_context_builder is not None else self._build_context(evidence)
            if not context:
                return self.NO_KNOWLEDGE_MESSAGE

            if self.answer_extractor is None:
                self.conversation.add_user_message(user_message)
                self.conversation.add_assistant_message(self.UNSUPPORTED_ANSWER_MESSAGE)
                return self.UNSUPPORTED_ANSWER_MESSAGE

            if aspect_queries:
                candidates = self.answer_extractor.extract_candidates(
                    question=user_message,
                    context=context,
                    aspect_queries=aspect_queries,
                )
            else:
                candidates = self.answer_extractor.extract_candidates(
                    question=user_message,
                    context=context,
                )

            if not candidates:
                return self.UNSUPPORTED_ANSWER_MESSAGE

            selection = self._select_answers(candidates, context)
            texts = self._selection_texts(selection)
            if not texts:
                if self.answerability_checker is not None and getattr(selection, "rejected", None):
                    assessment = self.answerability_checker.assess(user_message, selection)
                    self.last_assessment = assessment
                    self.last_selection = selection
                    return self.UNSUPPORTED_ANSWER_MESSAGE
                return self.NO_KNOWLEDGE_MESSAGE if not getattr(selection, "rejected", None) else self.UNSUPPORTED_ANSWER_MESSAGE
            if self._maybe_reject_irrelevant_selection(selection):
                return self.UNSUPPORTED_ANSWER_MESSAGE
            if self.answerability_checker is not None:
                assessment = self.answerability_checker.assess(user_message, selection)
                self.last_assessment = assessment
                if not self._assessment_is_answerable(assessment):
                    return self.UNSUPPORTED_ANSWER_MESSAGE

            self.last_selection = selection
            self.last_context = context
            self.last_documents = evidence

            answer = self._selected_answer(selection)
            if answer is None:
                return self.NO_KNOWLEDGE_MESSAGE
            return answer["text"]

        if self.knowledge_base is not None:
            queries = [user_message]
            aspect_queries = []
            if self.query_decomposer is not None:
                decomposed = self._extract_decomposition_queries(user_message)
                if decomposed:
                    aspect_queries = []
                    for item in decomposed:
                        aspect = item.get("aspect") if isinstance(item, dict) else None
                        question_text = item.get("question") if isinstance(item, dict) else None
                        if not isinstance(aspect, str) or not aspect.strip():
                            continue
                        if not isinstance(question_text, str) or not question_text.strip():
                            continue
                        normalized_aspect = aspect.strip()
                        if normalized_aspect.lower() == "original":
                            continue
                        aspect_queries.append({
                            "aspect": normalized_aspect,
                            "question": question_text.strip(),
                        })
                    queries = [
                        item.get("question") if isinstance(item, dict) else item
                        for item in decomposed
                        if isinstance(item, dict) and isinstance(item.get("question"), str) and item["question"].strip()
                    ] or [user_message]
                else:
                    queries = [user_message]

            documents = []
            seen = set()
            for query in queries:
                if not isinstance(query, str) or not query.strip():
                    continue
                result = self.knowledge_base.search(query=query.strip(), top_k=self.retrieval_top_k)
                for document in result or []:
                    document_id = None
                    if isinstance(document, dict):
                        document_id = document.get("id")
                    elif hasattr(document, "id"):
                        document_id = getattr(document, "id")
                    if document_id is not None and document_id in seen:
                        continue
                    if document_id is not None:
                        seen.add(document_id)
                    documents.append(document)

            context = self._build_context(documents)
            if not context:
                return self.NO_KNOWLEDGE_MESSAGE

            if self.context_validator is not None and not self.context_validator.is_relevant(question=user_message, context=context):
                return self.NO_KNOWLEDGE_MESSAGE

            if self.answer_extractor is None:
                self.conversation.add_user_message(user_message)
                self.conversation.add_assistant_message(self.UNSUPPORTED_ANSWER_MESSAGE)
                return self.UNSUPPORTED_ANSWER_MESSAGE

            candidate_kwargs = {"question": user_message, "context": context}
            if aspect_queries:
                candidate_kwargs["aspect_queries"] = aspect_queries
            candidates = self.answer_extractor.extract_candidates(**candidate_kwargs)
            if not candidates:
                return self.UNSUPPORTED_ANSWER_MESSAGE

            selection = self._select_answers(candidates, context)
            texts = self._selection_texts(selection)
            if not texts:
                if self.answerability_checker is not None and getattr(selection, "rejected", None):
                    assessment = self.answerability_checker.assess(user_message, selection)
                    self.last_assessment = assessment
                    self.last_selection = selection
                    return self.UNSUPPORTED_ANSWER_MESSAGE
                return self.UNSUPPORTED_ANSWER_MESSAGE
            if self._maybe_reject_irrelevant_selection(selection):
                return self.UNSUPPORTED_ANSWER_MESSAGE

            if self.answerability_checker is not None:
                assessment = self.answerability_checker.assess(user_message, selection)
                self.last_assessment = assessment
                if not self._assessment_is_answerable(assessment):
                    return self.UNSUPPORTED_ANSWER_MESSAGE

            self.last_selection = selection
            self.last_context = context
            self.last_documents = documents

            answer = self._selected_answer(selection)
            if answer is None:
                return self.NO_KNOWLEDGE_MESSAGE
            return answer["text"]

        self.conversation.add_user_message(user_message)
        history = self.conversation.get_history()
        prompt = self.prompt_builder.build(
            history=history,
            user_message=user_message,
        )
        response = self.generator.generate(prompt)
        self.conversation.add_assistant_message(response)
        return response



