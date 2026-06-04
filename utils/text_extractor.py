"""
TextExtractor — synchronous text extraction from .pdf and .docx files.

PDF  : pdfplumber (handles multi-column layouts and embedded fonts better than PyPDF2)
DOCX : python-docx with in-order block iteration (paragraphs + table cells)
       — no LibreOffice conversion required
"""

from pathlib import Path

import pdfplumber
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph


class TextExtractor:
    SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

    def extract(self, file_path: str) -> str:
        """Dispatch to the correct extractor based on the file extension."""
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            return self.extract_pdf(file_path)
        if suffix == ".docx":
            return self.extract_docx(file_path)
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
        )

    def extract_pdf(self, file_path: str) -> str:
        """Extract text page-by-page using pdfplumber."""
        pages: list[str] = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and text.strip():
                    pages.append(text.strip())
        return "\n\n".join(pages)

    def extract_docx(self, file_path: str) -> str:
        """
        Extract text from a DOCX in document order.

        Iterates the raw XML body so paragraphs and tables appear in the same
        sequence as in the original document, preserving narrative flow.
        Table rows are joined with ' | ' so columns stay readable.
        """
        doc = Document(file_path)
        parts: list[str] = []
        for block in self._iter_blocks(doc):
            if isinstance(block, Paragraph):
                if block.text.strip():
                    parts.append(block.text.strip())
            elif isinstance(block, Table):
                for row in block.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        parts.append(" | ".join(cells))
        return "\n".join(parts)

    @staticmethod
    def _iter_blocks(doc: Document):
        """Yield Paragraph and Table objects from the document body in order."""
        for child in doc.element.body:
            tag = child.tag
            if tag == qn("w:p"):
                yield Paragraph(child, doc)
            elif tag == qn("w:tbl"):
                yield Table(child, doc)
