import json
import pytest
from unittest.mock import Mock

from cognexa.knowledge.query.question_decomposer import QuestionDecomposer
from cognexa.knowledge.query.query_analyzer import QueryAnalyzer


def make_response(*items):
    return json.dumps({"sub_questions": list(items)})


def make_sub_question(aspect, question):
    return {
        "aspect": aspect,
        "question": question,
    }


class TestQuestionDecomposer:

    def setup_method(self):
        self.generator = Mock()
        self.analyzer = Mock(spec=QueryAnalyzer)
        self.analyzer.classify.return_value = "multi_aspect"

        self.decomposer = QuestionDecomposer(
            generator=self.generator,
            analyzer=self.analyzer,
        )

    def test_simple_question_does_not_call_llm(self):
        question = "What is RAG?"
        self.analyzer.classify.return_value = "simple"

        result = self.decomposer.decompose(question)

        assert result == [question]
        self.generator.generate.assert_not_called()

    def test_complex_question_is_decomposed(self):
        question = "What is RAG, and how does it work?"

        self.generator.generate.return_value = make_response(
            make_sub_question("definition", "What is RAG?"),
            make_sub_question("process", "How does RAG work?"),
        )

        result = self.decomposer.decompose(question)

        assert result == [
            question,
            "What is RAG?",
            "How does RAG work?",
        ]
        self.generator.generate.assert_called_once()

    def test_duplicate_questions_are_removed(self):
        question = "What is RAG, and how does it work?"

        self.generator.generate.return_value = make_response(
            make_sub_question("definition", "What is RAG?"),
            make_sub_question("definition", "what is rag?"),
            make_sub_question("process", "How does RAG work?"),
        )

        result = self.decomposer.decompose(question)

        assert result == [
            question,
            "What is RAG?",
            "How does RAG work?",
        ]

    def test_empty_questions_are_removed(self):
        question = "What is RAG, and how does it work?"

        self.generator.generate.return_value = make_response(
            make_sub_question("definition", "What is RAG?"),
            make_sub_question("definition", ""),
            make_sub_question("process", "   "),
            make_sub_question("process", "How does RAG work?"),
        )

        result = self.decomposer.decompose(question)

        assert result == [
            question,
            "What is RAG?",
            "How does RAG work?",
        ]

    def test_invalid_json_falls_back_to_original(self):
        question = "What is RAG, and how does it work?"
        self.generator.generate.return_value = "invalid JSON"

        result = self.decomposer.decompose(question)

        assert result == [question]

    def test_missing_sub_questions_field_falls_back(self):
        question = "What is RAG, and how does it work?"

        self.generator.generate.return_value = json.dumps({
            "questions": [
                "What is RAG?",
                "How does RAG work?",
            ]
        })

        result = self.decomposer.decompose(question)

        assert result == [question]

    def test_sub_questions_must_be_a_list(self):
        question = "What is RAG, and how does it work?"

        self.generator.generate.return_value = json.dumps({
            "sub_questions": "What is RAG?"
        })

        result = self.decomposer.decompose(question)

        assert result == [question]

    def test_sub_question_must_be_an_object(self):
        question = "What is RAG, and how does it work?"

        self.generator.generate.return_value = json.dumps({
            "sub_questions": [
                "What is RAG?",
                "How does RAG work?",
            ]
        })

        result = self.decomposer.decompose(question)

        assert result == [question]

    def test_sub_question_requires_aspect(self):
        question = "What is RAG, and how does it work?"

        self.generator.generate.return_value = make_response(
            {"question": "What is RAG?"},
            make_sub_question("process", "How does RAG work?"),
        )

        result = self.decomposer.decompose(question)

        assert result == [question]

    def test_sub_question_requires_question(self):
        question = "What is RAG, and how does it work?"

        self.generator.generate.return_value = make_response(
            {"aspect": "definition"},
            make_sub_question("process", "How does RAG work?"),
        )

        result = self.decomposer.decompose(question)

        assert result == [question]

    def test_non_string_aspect_falls_back(self):
        question = "What is RAG, and how does it work?"

        self.generator.generate.return_value = make_response(
            make_sub_question(123, "What is RAG?"),
            make_sub_question("process", "How does RAG work?"),
        )

        result = self.decomposer.decompose(question)

        assert result == [question]

    def test_non_string_question_falls_back(self):
        question = "What is RAG, and how does it work?"

        self.generator.generate.return_value = make_response(
            make_sub_question("definition", 123),
            make_sub_question("process", "How does RAG work?"),
        )

        result = self.decomposer.decompose(question)

        assert result == [question]

    def test_empty_sub_questions_falls_back(self):
        question = "What is RAG, and how does it work?"
        self.generator.generate.return_value = make_response()

        result = self.decomposer.decompose(question)

        assert result == [question]

    def test_empty_question_raises_value_error(self):
        with pytest.raises(ValueError, match="question cannot be empty"):
            self.decomposer.decompose("   ")

        self.generator.generate.assert_not_called()

    def test_non_string_question_raises_type_error(self):
        with pytest.raises(TypeError, match="question must be a string"):
            self.decomposer.decompose(None)

        self.generator.generate.assert_not_called()

    def test_comparison_missing_subject_falls_back(self):
        question = "Compare RAG and fine-tuning."

        self.analyzer.classify.return_value = "comparison"
        self.generator.generate.return_value = make_response(
            make_sub_question("rag", "What is RAG?"),
            make_sub_question(
                "comparison",
                "Compare RAG with vector databases.",
            ),
        )

        result = self.decomposer.decompose(question)

        assert result == [question]

    def test_comparison_without_comparison_aspect_falls_back(self):
        question = "Compare RAG and fine-tuning."

        self.analyzer.classify.return_value = "comparison"
        self.generator.generate.return_value = make_response(
            make_sub_question("rag", "What is RAG?"),
            make_sub_question(
                "fine_tuning",
                "What is fine-tuning?",
            ),
        )

        result = self.decomposer.decompose(question)

        assert result == [question]

    def test_valid_comparison_is_accepted(self):
        question = "Compare RAG and fine-tuning."

        self.analyzer.classify.return_value = "comparison"
        self.generator.generate.return_value = make_response(
            make_sub_question("rag", "What is RAG?"),
            make_sub_question(
                "fine_tuning",
                "What is fine-tuning?",
            ),
            make_sub_question(
                "comparison",
                "How do RAG and fine-tuning compare?",
            ),
        )

        result = self.decomposer.decompose(question)

        assert result[0] == question
        assert len(result) == 4

    def test_cause_effect_missing_consequences_falls_back(self):
        question = "What are the causes and consequences of overfitting?"

        self.analyzer.classify.return_value = "cause_effect"
        self.generator.generate.return_value = make_response(
            make_sub_question(
                "causes",
                "What factors contribute to overfitting?",
            ),
        )

        result = self.decomposer.decompose(question)

        assert result == [question]

    def test_valid_cause_effect_is_accepted(self):
        question = "What are the causes and consequences of overfitting?"

        self.analyzer.classify.return_value = "cause_effect"
        self.generator.generate.return_value = make_response(
            make_sub_question(
                "causes",
                "What factors contribute to overfitting?",
            ),
            make_sub_question(
                "consequences",
                "How does overfitting affect model generalization?",
            ),
        )

        result = self.decomposer.decompose(question)

        assert result[0] == question
        assert len(result) == 3

    def test_multi_aspect_missing_process_falls_back(self):
        question = "What is RAG, and how does it work?"

        self.generator.generate.return_value = make_response(
            make_sub_question("definition", "What is RAG?"),
            make_sub_question("benefits", "Why is RAG useful?"),
        )

        result = self.decomposer.decompose(question)

        assert result == [question]

    def test_relationship_missing_relationship_aspect_falls_back(self):
        question = "What is the relationship between cats and dogs?"

        self.analyzer.classify.return_value = "relationship"
        self.generator.generate.return_value = make_response(
            make_sub_question("cats", "What are cats?"),
            make_sub_question("dogs", "What are dogs?"),
        )

        result = self.decomposer.decompose(question)

        assert result == [question]

    def test_valid_relationship_is_accepted(self):
        question = "What is the relationship between cats and dogs?"

        self.analyzer.classify.return_value = "relationship"
        self.generator.generate.return_value = make_response(
            make_sub_question(
                "shared_origin",
                "Do cats and dogs share a common ancestor?",
            ),
            make_sub_question(
                "differences",
                "How do cats and dogs differ genetically?",
            ),
        )

        result = self.decomposer.decompose(question)

        assert result[0] == question
        assert len(result) == 3

    def test_prevention_missing_solution_questions_falls_back(self):
        question = "How can overfitting be reduced?"

        self.analyzer.classify.return_value = "prevention"
        self.generator.generate.return_value = make_response(
            make_sub_question(
                "definition",
                "What is overfitting in machine learning?",
            ),
            make_sub_question(
                "causes",
                "What causes overfitting in machine learning?",
            ),
        )

        result = self.decomposer.decompose(question)

        assert result == [question]

    def test_valid_prevention_is_accepted(self):
        question = "How can overfitting be reduced?"

        self.analyzer.classify.return_value = "prevention"
        self.generator.generate.return_value = make_response(
            make_sub_question(
                "regularization",
                "How can regularization reduce overfitting?",
            ),
            make_sub_question(
                "early_stopping",
                "How can early stopping help prevent overfitting?",
            ),
        )

        result = self.decomposer.decompose(question)

        assert result[0] == question
        assert len(result) == 3

    def test_list_with_duplicate_aspects_falls_back(self):
        question = (
            "What are other factors that contribute to overfitting?"
        )

        self.analyzer.classify.return_value = "list"
        self.generator.generate.return_value = make_response(
            make_sub_question(
                "data_related",
                "How can insufficient training data contribute to overfitting?",
            ),
            make_sub_question(
                "data_related",
                "How can noisy labels contribute to overfitting?",
            ),
        )

        result = self.decomposer.decompose(question)

        assert result == [question]

    def test_valid_list_is_accepted(self):
        question = (
            "What are other factors that contribute to overfitting?"
        )

        self.analyzer.classify.return_value = "list"
        self.generator.generate.return_value = make_response(
            make_sub_question(
                "data_related",
                "How can insufficient training data contribute to overfitting?",
            ),
            make_sub_question(
                "model_related",
                "How can excessive model complexity contribute to overfitting?",
            ),
            make_sub_question(
                "training_related",
                "How can training for too many epochs contribute to overfitting?",
            ),
        )

        result = self.decomposer.decompose(question)

        assert result[0] == question
        assert len(result) == 4

