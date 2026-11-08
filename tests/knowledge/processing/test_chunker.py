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

            assert (
                current_chunk[-5:] == next_chunk[:5]
            )

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
