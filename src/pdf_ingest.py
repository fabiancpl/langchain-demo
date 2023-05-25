from typing import Callable, List, Tuple, Dict

import os
import re

import PyPDF4
import pdfplumber

from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


class PDFIngestor:
    def __init__(self):
        self.file_path = "data/pnd-2023.pdf"
        self.vector_dir = "data/chroma"
        self.collection_name = "pnd-2023"

    def _extract_metadata_from_pdf(self, file_path: str) -> dict:
        with open(file_path, "rb") as pdf_file:
            reader = PyPDF4.PdfFileReader(pdf_file)
            metadata = reader.getDocumentInfo()
            return {
                "title": metadata.get("/Title", "").strip(),
                "author": metadata.get("/Author", "").strip(),
                "creation_date": metadata.get("/CreationDate", "").strip(),
            }

    def _extract_pages_from_pdf(self, file_path: str) -> List[Tuple[int, str]]:
        """
        Extracts the text from each page of the PDF.

        :param file_path: The path to the PDF file.
        :return: A list of tuples containing the page number and the extracted text.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with pdfplumber.open(file_path) as pdf:
            pages = []
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text.strip():  # Check if extracted text is not empty
                    pages.append((page_num + 1, text))
        return pages

    def _parse_pdf(self, file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
        """
        Extracts the title and text from each page of the PDF.

        :param file_path: The path to the PDF file.
        :return: A tuple containing the title and a list of tuples with page numbers and extracted text.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        metadata = self._extract_metadata_from_pdf(file_path)
        pages = self._extract_pages_from_pdf(file_path)

        return pages, metadata

    def _merge_hyphenated_words(self, text: str) -> str:
        return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    def _fix_newlines(self, text: str) -> str:
        return re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    def _remove_multiple_newlines(self, text: str) -> str:
        return re.sub(r"\n{2,}", "\n", text)

    def _clean_text(
        self, 
        pages: List[Tuple[int, str]], cleaning_functions: List[Callable[[str], str]]
    ) -> List[Tuple[int, str]]:
        cleaned_pages = []
        for page_num, text in pages:
            for cleaning_function in cleaning_functions:
                text = cleaning_function(text)
            cleaned_pages.append((page_num, text))
        return cleaned_pages

    def _text_to_docs(self, text: List[str], metadata: Dict[str, str]) -> List[Document]:
        """Converts list of strings to a list of Documents with metadata."""
        doc_chunks = []

        for page_num, page in text:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                chunk_overlap=200,
            )
            chunks = text_splitter.split_text(page)
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "page_number": page_num,
                        "chunk": i,
                        "source": f"p{page_num}-{i}",
                        **metadata,
                    },
                )
                doc_chunks.append(doc)

        return doc_chunks

    def ingest(self):
        if not os.path.exists(self.vector_dir):
            # Loading the PDF document
            
            raw_pages, metadata = self._parse_pdf(self.file_path)
            
            # Splitting the document in chunks
            cleaning_functions = [
                self._merge_hyphenated_words,
                self._fix_newlines,
                self._remove_multiple_newlines,
            ]
            cleaned_text_pdf = self._clean_text(raw_pages, cleaning_functions)
            document_chunks = self._text_to_docs(cleaned_text_pdf, metadata)

            # Generating embeddings and store them in the vector DB
            embeddings = OpenAIEmbeddings()
            vector_store = Chroma.from_documents(
                document_chunks,
                embeddings,
                collection_name=self.collection_name,
                persist_directory=self.vector_dir,
            )
            vector_store.persist()