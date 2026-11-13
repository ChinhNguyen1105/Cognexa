
import pytest

from cognexa.knowledge.query.query_analyzer import QueryAnalyzer


class TestQueryAnalyzer:

    def setup_method(self):
        self.analyzer = QueryAnalyzer()

    @pytest.mark.parametrize(
        "query, expected",
        [
            ("Compare RAG and fine-tuning.", "comparison"),
            (
                "What are the causes and consequences of overfitting?",
                "cause_effect",
            ),
            (
                "What is RAG, and how does it work?",
                "multi_aspect",
            ),
            (
                "What is the relationship between cats and dogs?",
                "relationship",
            ),
            (
                "How can overfitting be reduced?",
                "prevention",
            ),
            (
                "What methods can prevent overfitting?",
                "prevention",
            ),
            (
                "How can we mitigate overfitting?",
                "prevention",
            ),
            (
                "What are other factors that contribute to overfitting?",
                "list",
            ),
            (
                "List the common causes of overfitting.",
                "list",
            ),
            ("What is RAG?", "simple"),
        ],
    )
    def test_classify_question(self, query, expected):
        assert self.analyzer.classify(query) == expected

    @pytest.mark.parametrize(
        "query",
        [
            "What is RAG?",
            "Explain how embeddings work.",
            "What is the purpose of a vector database?",
        ],
    )
    def test_simple_query(self, query):
        assert self.analyzer.is_complex(query) is False

    @pytest.mark.parametrize(
        "query",
        [
            "Compare RAG and fine-tuning.",
            "What are the advantages and disadvantages of RAG.",
            "Explain the causes and consequences of overfitting.",
            "What is RAG, and how does it work?",
            "How does RAG work, and why is it useful?",
            "What is RAG? How does it work?",
            "So sánh RAG và fine-tuning.",
            "Phân tích ưu điểm và nhược điểm của RAG.",
            "Nguyên nhân và hậu quả của overfitting là gì?",
            "How can overfitting be reduced?",
            "What methods can prevent overfitting?",
            "What are other factors that contribute to overfitting?",
            "List the common causes of overfitting.",
        ],
    )
    def test_complex_query(self, query):
        assert self.analyzer.is_complex(query) is True

    def test_relationship_query_is_complex(self):
        query = "What is the relationship between cats and dogs?"

        assert self.analyzer.is_complex(query) is True

    def test_query_with_multiple_question_marks(self):
        query = "What is RAG? How does it work?"

        assert self.analyzer.is_complex(query) is True

    def test_query_with_leading_and_trailing_spaces(self):
        assert self.analyzer.is_complex("  What is RAG?  ") is False

    def test_empty_query_raises_value_error(self):
        with pytest.raises(ValueError, match="query cannot be empty"):
            self.analyzer.is_complex("   ")

    def test_non_string_query_raises_type_error(self):
        with pytest.raises(TypeError, match="query must be a string"):
            self.analyzer.is_complex(None)
