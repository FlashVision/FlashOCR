"""PDF document processing for FlashOCR.

Extracts text, images, and tables from PDF documents.
Supports multiple backends: pdfplumber (preferred) and PyPDF2 (fallback).
"""

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

_PDFPLUMBER_AVAILABLE = False
_PYPDF2_AVAILABLE = False

try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    pass

try:
    import PyPDF2
    _PYPDF2_AVAILABLE = True
except ImportError:
    pass


@dataclass
class PDFImage:
    """An image extracted from a PDF."""

    page_number: int
    image_data: bytes
    width: int = 0
    height: int = 0
    format: str = ""
    bbox: Optional[Tuple[float, float, float, float]] = None

    def to_numpy(self) -> Optional[np.ndarray]:
        """Decode image bytes to a numpy array (requires PIL)."""
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(self.image_data))
            return np.array(img)
        except Exception:
            return None


@dataclass
class PDFTable:
    """A table extracted from a PDF."""

    page_number: int
    data: List[List[Optional[str]]] = field(default_factory=list)
    bbox: Optional[Tuple[float, float, float, float]] = None
    num_rows: int = 0
    num_cols: int = 0

    def to_csv(self, delimiter: str = ",") -> str:
        rows = []
        for row in self.data:
            cells = [str(c) if c is not None else "" for c in row]
            rows.append(delimiter.join(cells))
        return "\n".join(rows)

    def to_html(self) -> str:
        rows_html = []
        for row in self.data:
            cells = "".join(f"<td>{c if c else ''}</td>" for c in row)
            rows_html.append(f"<tr>{cells}</tr>")
        return f"<table>{''.join(rows_html)}</table>"


@dataclass
class PDFPage:
    """Processed content from a single PDF page."""

    page_number: int
    text: str = ""
    tables: List[PDFTable] = field(default_factory=list)
    images: List[PDFImage] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0


@dataclass
class PDFDocument:
    """A fully processed PDF document."""

    path: str = ""
    num_pages: int = 0
    pages: List[PDFPage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)

    @property
    def all_tables(self) -> List[PDFTable]:
        return [t for p in self.pages for t in p.tables]

    @property
    def all_images(self) -> List[PDFImage]:
        return [img for p in self.pages for img in p.images]


class PDFProcessor:
    """Extract text, images, and tables from PDF documents.

    Uses pdfplumber when available (best table extraction), falling back
    to PyPDF2 for basic text/image extraction.

    Args:
        backend: ``"auto"``, ``"pdfplumber"``, or ``"pypdf2"``.
        extract_images: Whether to extract embedded images.
        extract_tables: Whether to extract tables.
        table_settings: pdfplumber table extraction settings.
        password: PDF password for encrypted documents.
    """

    def __init__(
        self,
        backend: str = "auto",
        extract_images: bool = True,
        extract_tables: bool = True,
        table_settings: Optional[Dict] = None,
        password: Optional[str] = None,
    ):
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.table_settings = table_settings or {}
        self.password = password

        if backend == "auto":
            if _PDFPLUMBER_AVAILABLE:
                self.backend = "pdfplumber"
            elif _PYPDF2_AVAILABLE:
                self.backend = "pypdf2"
            else:
                raise ImportError(
                    "No PDF backend available. Install pdfplumber or PyPDF2: "
                    "pip install pdfplumber PyPDF2"
                )
        else:
            self.backend = backend

    def process(self, pdf_path: Union[str, Path]) -> PDFDocument:
        """Process a PDF file and extract all content.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ``PDFDocument`` with text, tables, and images per page.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if self.backend == "pdfplumber":
            return self._process_pdfplumber(pdf_path)
        return self._process_pypdf2(pdf_path)

    def process_bytes(self, pdf_bytes: bytes) -> PDFDocument:
        """Process PDF from raw bytes."""
        if self.backend == "pdfplumber":
            return self._process_pdfplumber_stream(io.BytesIO(pdf_bytes))
        return self._process_pypdf2_stream(io.BytesIO(pdf_bytes))

    def extract_text(self, pdf_path: Union[str, Path]) -> str:
        """Quick text-only extraction from a PDF."""
        doc = self.process(pdf_path)
        return doc.full_text

    def extract_page_images(
        self,
        pdf_path: Union[str, Path],
        dpi: int = 200,
    ) -> List[np.ndarray]:
        """Render PDF pages as images (requires pdf2image or fitz).

        Args:
            pdf_path: Path to PDF.
            dpi: Resolution for rendering.

        Returns:
            List of numpy arrays (H, W, 3) for each page.
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(pdf_path))
            images = []
            scale = dpi / 72.0
            mat = fitz.Matrix(scale, scale)
            for page in doc:
                pix = page.get_pixmap(matrix=mat)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                if pix.n == 4:
                    img = img[:, :, :3]
                images.append(img.copy())
            doc.close()
            return images
        except ImportError:
            pass

        try:
            from pdf2image import convert_from_path
            pil_images = convert_from_path(str(pdf_path), dpi=dpi)
            return [np.array(img) for img in pil_images]
        except ImportError:
            raise ImportError(
                "Page rendering requires PyMuPDF or pdf2image: "
                "pip install PyMuPDF  # or  pip install pdf2image"
            )

    # ── pdfplumber backend ──────────────────────────────────────────────

    def _process_pdfplumber(self, pdf_path: Path) -> PDFDocument:
        with pdfplumber.open(str(pdf_path), password=self.password) as pdf:
            return self._parse_pdfplumber(pdf, str(pdf_path))

    def _process_pdfplumber_stream(self, stream: io.BytesIO) -> PDFDocument:
        with pdfplumber.open(stream, password=self.password) as pdf:
            return self._parse_pdfplumber(pdf, "<bytes>")

    def _parse_pdfplumber(self, pdf, source: str) -> PDFDocument:
        pages = []
        for i, page in enumerate(pdf.pages):
            pdf_page = PDFPage(
                page_number=i + 1,
                width=float(page.width),
                height=float(page.height),
            )

            text = page.extract_text() or ""
            pdf_page.text = text

            if self.extract_tables:
                raw_tables = page.extract_tables(self.table_settings) or []
                for raw in raw_tables:
                    if not raw:
                        continue
                    table = PDFTable(
                        page_number=i + 1,
                        data=raw,
                        num_rows=len(raw),
                        num_cols=max(len(r) for r in raw) if raw else 0,
                    )
                    pdf_page.tables.append(table)

            if self.extract_images:
                for img_info in page.images:
                    try:
                        bbox = (
                            float(img_info.get("x0", 0)),
                            float(img_info.get("top", 0)),
                            float(img_info.get("x1", 0)),
                            float(img_info.get("bottom", 0)),
                        )
                        pdf_image = PDFImage(
                            page_number=i + 1,
                            image_data=b"",
                            width=int(bbox[2] - bbox[0]),
                            height=int(bbox[3] - bbox[1]),
                            bbox=bbox,
                        )
                        pdf_page.images.append(pdf_image)
                    except Exception as e:
                        logger.debug("Skipping image on page %d: %s", i + 1, e)

            pages.append(pdf_page)

        metadata = {}
        if hasattr(pdf, "metadata") and pdf.metadata:
            metadata = {k: str(v) for k, v in pdf.metadata.items() if v}

        return PDFDocument(
            path=source,
            num_pages=len(pages),
            pages=pages,
            metadata=metadata,
        )

    # ── PyPDF2 backend ──────────────────────────────────────────────────

    def _process_pypdf2(self, pdf_path: Path) -> PDFDocument:
        with open(pdf_path, "rb") as f:
            return self._parse_pypdf2(f, str(pdf_path))

    def _process_pypdf2_stream(self, stream: io.BytesIO) -> PDFDocument:
        return self._parse_pypdf2(stream, "<bytes>")

    def _parse_pypdf2(self, stream, source: str) -> PDFDocument:
        reader = PyPDF2.PdfReader(stream, password=self.password)

        metadata = {}
        if reader.metadata:
            for key in ("/Title", "/Author", "/Subject", "/Creator", "/Producer"):
                val = reader.metadata.get(key)
                if val:
                    metadata[key.lstrip("/")] = str(val)

        pages = []
        for i, page in enumerate(reader.pages):
            pdf_page = PDFPage(page_number=i + 1)

            try:
                pdf_page.text = page.extract_text() or ""
            except Exception as e:
                logger.warning("Text extraction failed on page %d: %s", i + 1, e)

            if self.extract_images:
                pdf_page.images = self._extract_pypdf2_images(page, i + 1)

            mediabox = page.mediabox
            if mediabox:
                pdf_page.width = float(mediabox.width)
                pdf_page.height = float(mediabox.height)

            pages.append(pdf_page)

        return PDFDocument(
            path=source,
            num_pages=len(pages),
            pages=pages,
            metadata=metadata,
        )

    @staticmethod
    def _extract_pypdf2_images(page, page_number: int) -> List[PDFImage]:
        images = []
        if "/XObject" not in (page.get("/Resources") or {}):
            return images

        x_objects = page["/Resources"]["/XObject"].get_object()
        for obj_name in x_objects:
            obj = x_objects[obj_name].get_object()
            if obj.get("/Subtype") == "/Image":
                try:
                    width = int(obj.get("/Width", 0))
                    height = int(obj.get("/Height", 0))
                    data = obj.get_data()
                    img_format = "raw"
                    filt = obj.get("/Filter")
                    if filt == "/DCTDecode":
                        img_format = "jpeg"
                    elif filt == "/FlateDecode":
                        img_format = "png"

                    images.append(PDFImage(
                        page_number=page_number,
                        image_data=data,
                        width=width,
                        height=height,
                        format=img_format,
                    ))
                except Exception as e:
                    logger.debug("Skipping image %s on page %d: %s", obj_name, page_number, e)

        return images
