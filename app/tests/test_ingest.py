import pytest

from app.core.config import Settings
from app.services.ingest import load_documents


def test_load_documents_txt(tmp_path: pytest.TempPathFactory) -> None:
    file_path = tmp_path / "test.txt"
    file_path.write_text("Hello world")

    Settings.DATA_DIR = tmp_path
    docs = load_documents()

    assert len(docs) == 1
    assert "Hello world" in docs[0].page_content


def test_load_documents_skip_other_formats(
    tmp_path: pytest.TempPathFactory, caplog: pytest.LogCaptureFixture
) -> None:
    file_path = tmp_path / "test.docx"
    file_path.write_text("Dummy data")

    Settings.DATA_DIR = tmp_path
    docs = load_documents()

    assert len(docs) == 0

    with caplog.at_level("WARNING"):
        docs = load_documents()
    assert "⚠ Skip other format" in caplog.text
