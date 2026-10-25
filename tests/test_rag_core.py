import os
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

TEST_TMP = Path(__file__).resolve().parents[1] / ".test-tmp"
TEST_TMP.mkdir(exist_ok=True)
tempfile.tempdir = str(TEST_TMP)

from langchain_core.documents import Document  # noqa: E402
from rag_core import (  # noqa: E402
    build_vectorstore,
    conversation_history,
    display_page,
    format_docs,
    load_file,
)


class MemoryUpload:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


class RagCoreTests(unittest.TestCase):
    def test_reads_text_and_markdown_as_utf8_with_bom_support(self):
        for file_name in ("notes.txt", "guide.md"):
            with self.subTest(file_name=file_name):
                docs = load_file(file_name, "\ufeffCafé notes".encode("utf-8"))
                self.assertEqual(len(docs), 1)
                self.assertEqual(docs[0].page_content, "Café notes")
                self.assertEqual(docs[0].metadata["source"], file_name)

    def test_reads_csv_rows_with_citation_metadata(self):
        docs = load_file("sales.csv", b"name,total\nAmina,12\nBilal,19\n")
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[1].metadata["row"], 3)
        self.assertIn("CSV columns: name | total", docs[0].page_content)
        self.assertIn("Bilal | 19", docs[1].page_content)

    def test_reads_json_and_reports_invalid_json(self):
        docs = load_file("config.json", b'{"mode":"research"}')
        self.assertIn('"mode": "research"', docs[0].page_content)
        self.assertEqual(docs[0].metadata["source"], "config.json")
        with self.assertRaisesRegex(ValueError, "broken.json is not valid JSON"):
            load_file("broken.json", b"{")

    def test_reads_docx_paragraphs_and_tables(self):
        fake_docx = ModuleType("docx")

        class FakeDocx:
            def __init__(self, stream):
                self.paragraphs = [SimpleNamespace(text="  Project notes  "), SimpleNamespace(text="")]
                cell = lambda text: SimpleNamespace(text=text)
                self.tables = [SimpleNamespace(rows=[SimpleNamespace(cells=[cell("Owner"), cell("Amina")])])]

        fake_docx.Document = FakeDocx
        with patch.dict("sys.modules", {"docx": fake_docx}):
            docs = load_file("brief.docx", b"docx test payload")
        self.assertEqual(docs[0].page_content, "Project notes\nOwner | Amina")
        self.assertEqual(docs[0].metadata["source"], "brief.docx")

    def test_reads_pdf_pages_and_sets_the_uploaded_name_as_source(self):
        fake_page = Document(page_content="A PDF passage", metadata={"page": 1, "source": "temporary.pdf"})
        with patch("rag_core.PyPDFLoader") as loader:
            loader.return_value.load.return_value = [fake_page]
            docs = load_file("handbook.pdf", b"%PDF-1.4 test")

        self.assertEqual(docs[0].metadata["source"], "handbook.pdf")
        self.assertEqual(docs[0].metadata["page"], 1)
        temporary_path = loader.call_args.args[0]
        self.assertFalse(os.path.exists(temporary_path))

    def test_indexes_multiple_files_without_making_an_api_request(self):
        uploads = [
            MemoryUpload("first.txt", b"First source passage."),
            MemoryUpload("second.md", b"Second source passage."),
        ]
        expected_store = object()
        with (
            patch("rag_core.OpenAIEmbeddings") as embeddings,
            patch("rag_core.FAISS.from_documents", return_value=expected_store) as index,
        ):
            store, chunks, pages, names = build_vectorstore(uploads, 500, 100, "test-key")

        self.assertIs(store, expected_store)
        self.assertEqual(pages, 2)
        self.assertEqual(names, ["first.txt", "second.md"])
        self.assertEqual(chunks, 2)
        self.assertEqual(
            {doc.metadata["source"] for doc in index.call_args.args[0]},
            {"first.txt", "second.md"},
        )
        embeddings.assert_called_once_with(model="text-embedding-3-small", api_key="test-key")

    def test_skips_empty_files_and_rejects_invalid_chunk_overlap(self):
        warnings = []
        with (
            patch("rag_core.OpenAIEmbeddings"),
            patch("rag_core.FAISS.from_documents", return_value=object()),
        ):
            _, _, _, names = build_vectorstore(
                [MemoryUpload("empty.txt", b"  "), MemoryUpload("valid.txt", b"Text")],
                500,
                100,
                "test-key",
                on_warning=warnings.append,
            )
        self.assertEqual(names, ["valid.txt"])
        self.assertEqual(warnings, ["No readable text found in empty.txt; skipped."])
        with self.assertRaisesRegex(ValueError, "overlap must be smaller"):
            build_vectorstore([], 500, 500, "test-key")

    def test_formats_page_and_row_citations_for_retrieval(self):
        docs = [
            Document(page_content="A handbook passage", metadata={"source": "handbook.pdf", "page": 3}),
            Document(page_content="A sales row", metadata={"source": "sales.csv", "page": 2, "row": 3}),
        ]
        self.assertEqual(display_page(docs[0].metadata), "4")
        self.assertEqual(display_page(docs[1].metadata), "row 3")
        context = format_docs(docs)
        self.assertIn("[Source: handbook.pdf | Page/location: 4]", context)
        self.assertIn("[Source: sales.csv | Page/location: row 3]", context)

    def test_keeps_recent_chat_history_in_langchain_format(self):
        messages = [
            {"role": "user" if index % 2 == 0 else "assistant", "content": str(index)}
            for index in range(12)
        ]
        messages.insert(3, {"role": "system", "content": "not included"})
        self.assertEqual(conversation_history(messages, limit=10), [
            (message["role"], message["content"])
            for message in messages[-10:]
            if message["role"] in ("user", "assistant")
        ])


if __name__ == "__main__":
    unittest.main()

