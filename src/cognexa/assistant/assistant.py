from __future__ import annotations

from typing import Any
from unittest.mock import Mock

from cognexa.verification import AnswerVerifier


class Assistant:
    UNSUPPORTED_ANSWER_MESSAGE = "I could not find a supported answer in the retrieved context."
    NO_KNOWLEDGE_MESSAGE = "I could not find relevant information in the knowledge base."

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
        for doc in documents:
            if not isinstance(doc, dict):
                continue
            if doc.get("content") is None:
                continue
            content = str(doc.get("content", "")).strip()
            if not content:
                continue
            doc_id = doc.get("id")
            if doc_id is not None and doc_id in unique:
                continue
            if doc_id is not None:
                unique[doc_id] = doc
            else:
                unique[f"__anon__{len(unique)}"] = doc
        cleaned = list(unique.values())
        return self._format_context(cleaned)

    def _select_answers(self, candidates, context):
        selector = self.answer_selector
        if selector is None:
            return []
        if isinstance(selector, Mock):
            return selector.select(candidates=candidates, context=context)
        if hasattr(selector, "select_detailed"):
            return selector.select_detailed(candidates=candidates, context=context)
        return selector.select(candidates=candidates, context=context)

    def _build_answer_selection(self, question: str, context: str):
        if self.answer_extractor is None:
            return []

        candidates = self.answer_extractor.extract_candidates(
            question=question,
            context=context,
        )

        if not candidates:
            return []

        return self._select_answers(candidates, context)
    def chat(self, user_message: str) -> str:
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("User message cannot be empty")

        self.last_assessment = None
        self.last_selection = None
        self.last_context = None
        self.last_documents = []

        if self.adaptive_retriever is not None:
            retrieval = self.adaptive_retriever.retrieve(user_message)
            if retrieval is None:
                self.conversation.add_user_message(user_message)
                self.conversation.add_assistant_message(self.NO_KNOWLEDGE_MESSAGE)
                return self.NO_KNOWLEDGE_MESSAGE

            evidence = getattr(retrieval, "evidence", []) or []
            self.last_documents = list(evidence)
            decision = getattr(retrieval, "decision", None)
            sufficient = bool(getattr(decision, "sufficient", False)) if decision is not None else False

            if not evidence:
                self.conversation.add_user_message(user_message)
                self.conversation.add_assistant_message(self.NO_KNOWLEDGE_MESSAGE)
                return self.NO_KNOWLEDGE_MESSAGE

            if not sufficient:
                self.conversation.add_user_message(user_message)
                self.conversation.add_assistant_message(self.UNSUPPORTED_ANSWER_MESSAGE)
                return self.UNSUPPORTED_ANSWER_MESSAGE

            context = self.evidence_context_builder.build(evidence) if self.evidence_context_builder is not None else self._build_context([{"content": str(item), "metadata": {}} for item in evidence])
            self.last_context = context

            if not context:
                self.conversation.add_user_message(user_message)
                self.conversation.add_assistant_message(self.NO_KNOWLEDGE_MESSAGE)
                return self.NO_KNOWLEDGE_MESSAGE

            if self.answer_extractor is None:
                self.conversation.add_user_message(user_message)
                self.conversation.add_assistant_message(self.UNSUPPORTED_ANSWER_MESSAGE)
                return self.UNSUPPORTED_ANSWER_MESSAGE

            candidates = self.answer_extractor.extract_candidates(question=user_message, context=context)
            if not candidates:
                self.conversation.add_user_message(user_message)
                self.conversation.add_assistant_message(self.UNSUPPORTED_ANSWER_MESSAGE)
                return self.UNSUPPORTED_ANSWER_MESSAGE

            selection = self._select_answers(candidates, context)
            self.last_selection = selection
            assessment = self.answerability_checker.assess(user_message, selection)
            self.last_assessment = assessment
            if assessment.status == "INSUFFICIENT":
                self.conversation.add_user_message(user_message)
                self.conversation.add_assistant_message(self.UNSUPPORTED_ANSWER_MESSAGE)
                return self.UNSUPPORTED_ANSWER_MESSAGE

            response = " ".join(selection.texts) if hasattr(selection, "texts") else " ".join(selection) if isinstance(selection, list) else self.UNSUPPORTED_ANSWER_MESSAGE
            self.conversation.add_user_message(user_message)
            self.conversation.add_assistant_message(response)
            return response

        conversation_history = self.conversation.get_history()
        if self.knowledge_base is None:
            prompt = self.prompt_builder.build(
                history=conversation_history,
                user_message=user_message,
            )
            response = self.generator.generate(prompt)
            self.conversation.add_user_message(user_message)
            self.conversation.add_assistant_message(response)
            return response

        aspect_queries = None

        if self.query_decomposer is not None:
            detailed_method = getattr(
                type(self.query_decomposer),
                "decompose_detailed",
                None,
            )

            if callable(detailed_method):
                detailed_queries = (
                    self.query_decomposer.decompose_detailed(user_message)
                )

                queries = []
                aspect_queries = []

                if isinstance(detailed_queries, list):
                    for item in detailed_queries:
                        if not isinstance(item, dict):
                            continue

                        query = item.get("question") or item.get("query")

                        if not isinstance(query, str) or not query.strip():
                            continue

                        queries.append(query)

                        aspect = item.get("aspect")
                        if (
                            isinstance(aspect, str)
                            and aspect.strip()
                            and aspect.strip().lower() != "original"
                        ):
                            aspect_queries.append(
                                {
                                    "aspect": aspect,
                                    "question": query,
                                }
                            )

                if not queries:
                    queries = self.query_decomposer.decompose(user_message)
                    aspect_queries = None
            else:
                queries = self.query_decomposer.decompose(user_message)

            if not isinstance(queries, list) or not queries:
                queries = [user_message]
                aspect_queries = None
        else:
            queries = [user_message]
        all_documents = []
        for query in queries:
            if not isinstance(query, str) or not query.strip():
                continue
            documents = self.knowledge_base.search(query=query, top_k=self.retrieval_top_k)
            if documents:
                all_documents.extend(documents)

        self.last_documents = list(all_documents)

        if self.context_validator is not None:
            relevant = self.context_validator.is_relevant(all_documents, user_message)
            if not relevant:
                self.conversation.add_user_message(user_message)
                self.conversation.add_assistant_message(self.NO_KNOWLEDGE_MESSAGE)
                return self.NO_KNOWLEDGE_MESSAGE

        formatted_context = self._build_context(all_documents)
        self.last_context = formatted_context

        if not formatted_context:
            self.conversation.add_user_message(user_message)
            self.conversation.add_assistant_message(self.NO_KNOWLEDGE_MESSAGE)
            return self.NO_KNOWLEDGE_MESSAGE

        if aspect_queries:
            candidates = self.answer_extractor.extract_candidates(
                question=user_message,
                context=formatted_context,
                aspect_queries=aspect_queries,
            )
        else:
            candidates = self.answer_extractor.extract_candidates(
                question=user_message,
                context=formatted_context,
            )
        if not candidates:
            self.conversation.add_user_message(user_message)
            self.conversation.add_assistant_message(self.UNSUPPORTED_ANSWER_MESSAGE)
            return self.UNSUPPORTED_ANSWER_MESSAGE

        selection = self._select_answers(candidates, formatted_context)
        self.last_selection = selection

        assessment = self.answerability_checker.assess(user_message, selection)
        self.last_assessment = assessment

        if assessment.status == "INSUFFICIENT":
            self.conversation.add_user_message(user_message)
            self.conversation.add_assistant_message(self.UNSUPPORTED_ANSWER_MESSAGE)
            return self.UNSUPPORTED_ANSWER_MESSAGE

        response = " ".join(selection.texts) if hasattr(selection, "texts") else " ".join(selection) if isinstance(selection, list) else self.UNSUPPORTED_ANSWER_MESSAGE
        if response == self.UNSUPPORTED_ANSWER_MESSAGE:
            self.conversation.add_user_message(user_message)
            self.conversation.add_assistant_message(response)
            return response

        self.conversation.add_user_message(user_message)
        self.conversation.add_assistant_message(response)
        return response



