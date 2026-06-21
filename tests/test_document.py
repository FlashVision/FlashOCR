"""Tests for document processing: table extraction, layout analysis."""

import numpy as np
import torch


def test_table_detection_net():
    from flashocr.document.table_extractor import TableDetectionNet

    net = TableDetectionNet(base_channels=16)
    net.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        mask = net(x)
    assert mask.shape == (1, 1, 256, 256)


def test_table_structure_net():
    from flashocr.document.table_extractor import TableStructureNet

    net = TableStructureNet(base_channels=16)
    net.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        row_logits, col_logits = net(x)
    assert row_logits.ndim == 1 or row_logits.ndim == 2
    assert col_logits.ndim == 1 or col_logits.ndim == 2


def test_table_cell_dataclass():
    from flashocr.document.table_extractor import TableCell, Table

    cell = TableCell(row=0, col=0, bbox=(10, 10, 50, 30), text="hello")
    assert cell.text == "hello"

    table = Table(
        bbox=(0, 0, 100, 100),
        num_rows=2,
        num_cols=2,
        cells=[
            TableCell(row=0, col=0, text="A"),
            TableCell(row=0, col=1, text="B"),
            TableCell(row=1, col=0, text="C"),
            TableCell(row=1, col=1, text="D"),
        ],
    )
    grid = table.to_grid()
    assert grid[0][0] == "A"
    assert grid[1][1] == "D"

    html = table.to_html()
    assert "<table>" in html
    assert "<td>A</td>" in html


def test_table_extractor_line_mode():
    from flashocr.document.table_extractor import TableExtractor

    extractor = TableExtractor(mode="line")
    image = np.ones((200, 300, 3), dtype=np.uint8) * 255
    tables = extractor.extract(image)
    assert isinstance(tables, list)


def test_layout_net_forward():
    from flashocr.document.layout_analyzer import LayoutNet

    net = LayoutNet(num_classes=9, base_ch=16, fpn_ch=32)
    net.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        heatmap, regression = net(x)
    assert heatmap.shape[1] == 9
    assert regression.shape[1] == 4


def test_layout_analyzer():
    from flashocr.document.layout_analyzer import LayoutAnalyzer, LayoutLabel

    analyzer = LayoutAnalyzer(score_threshold=0.99)
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    regions = analyzer.analyze(image)
    assert isinstance(regions, list)

    assert LayoutLabel.num_classes() == 9
    assert LayoutLabel.name_of(0) == "title"


def test_pdf_processor_init():
    from flashocr.document.pdf_processor import PDFDocument, PDFPage

    page = PDFPage(page_number=1, text="Hello World")
    assert page.text == "Hello World"

    doc = PDFDocument(path="/test.pdf", num_pages=1, pages=[page])
    assert doc.full_text == "Hello World"
    assert doc.num_pages == 1
