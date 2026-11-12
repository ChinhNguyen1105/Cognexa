
import pytest

from cognexa.knowledge.processing.chunker import TextChunker


class TestTextChunker:

    def test_chunk_short_text(self):
        # Arrange
        chunker = TextChunker(
            chunk_size=100,
            chunk_overlap=20,
        )
        text = "Cognexa is an AI project."

        # Act
        chunks = chunker.chunk_text(text)

        # Assert
        assert chunks == [text]

    def test_chunk_long_text(self):
        # Arrange
        chunker = TextChunker(
            chunk_size=20,
            chunk_overlap=5,
        )
        text = (
            "Cognexa is an AI learning project "
            "for understanding artificial intelligence."
        )

        # Act
        chunks = chunker.chunk_text(text)

        # Assert
        assert len(chunks) > 1

        for chunk in chunks:
            assert len(chunk) <= chunker.chunk_size

    def test_chunk_empty_text(self):
        # Arrange
        chunker = TextChunker()

        # Act
        chunks = chunker.chunk_text("")

        # Assert
        assert chunks == []

    def test_chunk_whitespace_only_text(self):
        # Arrange
        chunker = TextChunker()

        # Act
        chunks = chunker.chunk_text("   \n\t   ")

        # Assert
        assert chunks == []

    def test_chunk_trims_whitespace(self):
        # Arrange
        chunker = TextChunker()
        text = "   Cognexa AI project.   "

        # Act
        chunks = chunker.chunk_text(text)

        # Assert
        assert chunks == ["Cognexa AI project."]

    def test_chunk_overlap(self):
        # Arrange
        chunker = TextChunker(
            chunk_size=20,
            chunk_overlap=5,
        )
        text = "AAAAA BBBBB CCCCC DDDDD EEEEE FFFFF"

        # Act
        chunks = chunker.chunk_text(text)

        # Assert
        assert len(chunks) > 1

        for index in range(len(chunks) - 1):
            current_chunk = chunks[index]
            next_chunk = chunks[index + 1]

            assert current_chunk[-5:] == next_chunk[:5]

    def test_chunk_size_zero_raises_error(self):
        with pytest.raises(ValueError, match="chunk_size"):
            TextChunker(chunk_size=0)

    def test_negative_chunk_size_raises_error(self):
        with pytest.raises(ValueError, match="chunk_size"):
            TextChunker(chunk_size=-10)

    def test_negative_chunk_overlap_raises_error(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            TextChunker(
                chunk_size=100,
                chunk_overlap=-1,
            )

    def test_overlap_equal_to_chunk_size_raises_error(self):
        with pytest.raises(ValueError, match="smaller than"):
            TextChunker(
                chunk_size=100,
                chunk_overlap=100,
            )

    def test_overlap_greater_than_chunk_size_raises_error(self):
        with pytest.raises(ValueError, match="smaller than"):
            TextChunker(
                chunk_size=100,
                chunk_overlap=120,
            )

    def test_non_string_input_raises_error(self):
        # Arrange
        chunker = TextChunker()

        # Act & Assert
        with pytest.raises(TypeError, match="must be a string"):
            chunker.chunk_text(123)

    # ---------------------------------------------------------
    # List metadata tests
    # ---------------------------------------------------------

    def test_chunk_document_detects_list_in_single_child(self):
        # Arrange
        parent_chunker = TextChunker(
            chunk_size=500,
            chunk_overlap=0,
        )
        child_chunker = TextChunker(
            chunk_size=500,
            chunk_overlap=0,
        )
        text = (
            "Available tools:\n"
            "- Python\n"
            "- Git\n"
            "- Linux"
        )

        # Act
        result = parent_chunker.chunk_document(
            text=text,
            child_chunker=child_chunker,
        )

        # Assert
        assert len(result) == 1

        children = result[0]["children"]

        assert len(children) == 1
        assert children[0]["list_groups"] == [
            {
                "group_id": 0,
                "item_indices": [0, 1, 2],
            }
        ]

    def test_chunk_document_preserves_list_group_across_children(self):
        # Arrange
        parent_chunker = TextChunker(
            chunk_size=500,
            chunk_overlap=0,
        )
        child_chunker = TextChunker(
            chunk_size=40,
            chunk_overlap=0,
        )
        text = (
            "Available tools:\n"
            "- Python programming\n"
            "- Git version control\n"
            "- Linux operating system\n"
            "- Docker containerization\n"
            "- FastAPI web framework"
        )

        # Act
        result = parent_chunker.chunk_document(
            text=text,
            child_chunker=child_chunker,
        )

        # Assert
        children = result[0]["children"]

        assert len(children) > 1

        groups = [
            group
            for child in children
            for group in child["list_groups"]
        ]

        assert all(group["group_id"] == 0 for group in groups)

        retrieved_indices = {
            index
            for group in groups
            for index in group["item_indices"]
        }

        assert retrieved_indices == {0, 1, 2, 3, 4}

    def test_chunk_document_detects_independent_lists(self):
        # Arrange
        parent_chunker = TextChunker(
            chunk_size=1000,
            chunk_overlap=0,
        )
        child_chunker = TextChunker(
            chunk_size=1000,
            chunk_overlap=0,
        )
        text = (
            "Programming languages:\n"
            "- Python\n"
            "- Java\n"
            "\n"
            "Databases:\n"
            "- MySQL\n"
            "- PostgreSQL"
        )

        # Act
        result = parent_chunker.chunk_document(
            text=text,
            child_chunker=child_chunker,
        )

        # Assert
        children = result[0]["children"]

        assert len(children) == 1

        groups = children[0]["list_groups"]

        assert groups == [
            {
                "group_id": 0,
                "item_indices": [0, 1],
            },
            {
                "group_id": 1,
                "item_indices": [0, 1],
            },
        ]

    def test_chunk_document_does_not_detect_normal_prose_as_list(self):
        # Arrange
        parent_chunker = TextChunker(
            chunk_size=500,
            chunk_overlap=0,
        )
        child_chunker = TextChunker(
            chunk_size=500,
            chunk_overlap=0,
        )
        text = (
            "Python is a programming language.\n"
            "It supports object-oriented programming.\n"
            "It is widely used in AI development."
        )

        # Act
        result = parent_chunker.chunk_document(
            text=text,
            child_chunker=child_chunker,
        )

        # Assert
        children = result[0]["children"]

        assert len(children) == 1
        assert children[0]["list_groups"] == []
