#!/usr/bin/env python3
"""Export project Markdown reports to simple DOCX and PDF without pandoc.

The converter intentionally supports the subset used by the final reports:
headings, paragraphs, bullets, code fences, pipe tables, horizontal rules,
and images. Missing FILLME images are rendered as text placeholders.
"""
from __future__ import annotations

import html
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

REPORT_FONT_FAMILY = "Times New Roman"
PDF_FONT_REGULAR = "Times-Roman"
PDF_FONT_BOLD = "Times-Bold"
PDF_FONT_MONO = "Courier"


@dataclass
class Block:
    kind: str
    data: Any


def parse_markdown(text: str) -> list[Block]:
    lines = text.splitlines()
    blocks: list[Block] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped == "---":
            blocks.append(Block("hr", None))
            i += 1
            continue
        if stripped.startswith("```"):
            lang = stripped.strip("`").strip()
            code: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            blocks.append(Block("code", {"lang": lang, "text": "\n".join(code)}))
            continue
        m_img = re.match(r"!\[(.*?)\]\((.*?)\)\s*$", stripped)
        if m_img:
            blocks.append(Block("image", {"alt": m_img.group(1), "path": m_img.group(2)}))
            i += 1
            continue
        m_head = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m_head:
            blocks.append(Block("heading", {"level": len(m_head.group(1)), "text": m_head.group(2).strip()}))
            i += 1
            continue
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}:?", lines[i + 1]):
            table_lines = [lines[i], lines[i + 1]]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            rows = [_split_table_row(x) for x in table_lines]
            if len(rows) >= 2:
                blocks.append(Block("table", {"header": rows[0], "rows": rows[2:]}))
            continue
        if re.match(r"^\s*([-*])\s+", line) or re.match(r"^\s*\d+\.\s+", line):
            ordered = bool(re.match(r"^\s*\d+\.\s+", line))
            items: list[str] = []
            while i < len(lines):
                m_b = re.match(r"^\s*[-*]\s+(.*)$", lines[i])
                m_o = re.match(r"^\s*\d+\.\s+(.*)$", lines[i])
                if ordered and m_o:
                    items.append(m_o.group(1).strip())
                elif not ordered and m_b:
                    items.append(m_b.group(1).strip())
                else:
                    break
                i += 1
            blocks.append(Block("list", {"ordered": ordered, "items": items}))
            continue

        para = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt:
                break
            if nxt == "---" or nxt.startswith("```") or nxt.startswith("!") or nxt.startswith("#"):
                break
            if nxt.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}:?", lines[i + 1]):
                break
            if re.match(r"^\s*([-*])\s+", lines[i]) or re.match(r"^\s*\d+\.\s+", lines[i]):
                break
            para.append(nxt)
            i += 1
        blocks.append(Block("paragraph", " ".join(para)))
    return blocks


def _split_table_row(line: str) -> list[str]:
    line = line.strip().strip("|")
    return [cell.strip() for cell in line.split("|")]


def inline_to_reportlab(text: str) -> str:
    text = html.escape(text)
    text = (
        text.replace("&lt;u&gt;", "<u>")
        .replace("&lt;/u&gt;", "</u>")
        .replace("&lt;strong&gt;", "<b>")
        .replace("&lt;/strong&gt;", "</b>")
        .replace("&lt;b&gt;", "<b>")
        .replace("&lt;/b&gt;", "</b>")
    )
    text = re.sub(r"`([^`]+)`", rf"<font name='{PDF_FONT_MONO}'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def is_markdown_visible_caption(text: str) -> bool:
    """Captions are shown in Markdown/HTML but skipped in DOCX/PDF.

    DOCX/PDF already render image alt text as the figure caption, so the
    explicit HTML caption lines are only for the .md version.
    """
    stripped = text.strip()
    return bool(
        re.match(
            r'^<p\s+align="center"><em>(Figure\s+\d+\.|PANDA project icon\.).*</em></p>$',
            stripped,
        )
    )


def build_pdf(md_path: Path, out_path: Path) -> None:
    register_fonts()
    blocks = parse_markdown(md_path.read_text(encoding="utf-8"))
    styles = make_pdf_styles()
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        rightMargin=0.68 * inch,
        leftMargin=0.68 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title=md_path.stem,
    )
    story: list[Any] = []
    for block in blocks:
        k = block.kind
        d = block.data
        if k == "heading":
            level = min(d["level"], 4)
            story.append(Paragraph(inline_to_reportlab(d["text"]), styles[f"Heading{level}"]))
            story.append(Spacer(1, 6))
        elif k == "paragraph":
            if is_markdown_visible_caption(d):
                continue
            story.append(Paragraph(inline_to_reportlab(d), styles["Body"]))
            story.append(Spacer(1, 6))
        elif k == "list":
            for idx, item in enumerate(d["items"], 1):
                prefix = f"{idx}. " if d["ordered"] else "- "
                story.append(Paragraph(inline_to_reportlab(prefix + item), styles["ListBody"]))
            story.append(Spacer(1, 6))
        elif k == "code":
            story.append(Preformatted(d["text"], styles["Code"]))
            story.append(Spacer(1, 6))
        elif k == "table":
            story.append(pdf_table(d, styles))
            story.append(Spacer(1, 8))
        elif k == "image":
            flow = pdf_image(md_path.parent, d, styles, doc.width)
            story.extend(flow)
        elif k == "hr":
            story.append(Spacer(1, 10))
    doc.build(story)


def register_fonts() -> None:
    # ReportLab ships the standard PostScript Times family.  Using it keeps the
    # generated PDF in a Times New Roman-compatible academic style without
    # relying on a proprietary Times New Roman TTF being installed on Linux.
    pdfmetrics.registerFontFamily(
        "Times",
        normal=PDF_FONT_REGULAR,
        bold=PDF_FONT_BOLD,
        italic="Times-Italic",
        boldItalic="Times-BoldItalic",
    )


def make_pdf_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "Heading1": ParagraphStyle("Heading1", parent=base["Heading1"], fontName=PDF_FONT_BOLD, fontSize=18, leading=22, spaceBefore=10, spaceAfter=8),
        "Heading2": ParagraphStyle("Heading2", parent=base["Heading2"], fontName=PDF_FONT_BOLD, fontSize=15, leading=19, spaceBefore=8, spaceAfter=6),
        "Heading3": ParagraphStyle("Heading3", parent=base["Heading3"], fontName=PDF_FONT_BOLD, fontSize=12.5, leading=16, spaceBefore=6, spaceAfter=4),
        "Heading4": ParagraphStyle("Heading4", parent=base["Heading4"], fontName=PDF_FONT_BOLD, fontSize=11, leading=14, spaceBefore=5, spaceAfter=3),
        "Body": ParagraphStyle("Body", parent=base["BodyText"], fontName=PDF_FONT_REGULAR, fontSize=9.5, leading=13.2, alignment=TA_LEFT),
        "ListBody": ParagraphStyle("ListBody", parent=base["BodyText"], fontName=PDF_FONT_REGULAR, fontSize=9.5, leading=13.2, leftIndent=14, firstLineIndent=-8, alignment=TA_LEFT),
        "Caption": ParagraphStyle("Caption", parent=base["BodyText"], fontName=PDF_FONT_REGULAR, fontSize=8.5, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#475569")),
        "Code": ParagraphStyle("Code", parent=base["Code"], fontName=PDF_FONT_MONO, fontSize=7.8, leading=10, backColor=colors.HexColor("#f1f5f9"), borderPadding=5),
        "TableCell": ParagraphStyle("TableCell", parent=base["BodyText"], fontName=PDF_FONT_REGULAR, fontSize=7.2, leading=9.2),
        "TableHead": ParagraphStyle("TableHead", parent=base["BodyText"], fontName=PDF_FONT_BOLD, fontSize=7.2, leading=9.2, textColor=colors.white),
    }


def pdf_table(data: dict[str, Any], styles: dict[str, ParagraphStyle]) -> Table:
    rows = [[Paragraph(inline_to_reportlab(c), styles["TableHead"]) for c in data["header"]]]
    rows += [[Paragraph(inline_to_reportlab(c), styles["TableCell"]) for c in r] for r in data["rows"]]
    ncols = max(1, len(rows[0]))
    table = Table(rows, hAlign="LEFT", repeatRows=1, colWidths=[7.2 * inch / ncols] * ncols)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def pdf_image(base_dir: Path, data: dict[str, str], styles: dict[str, ParagraphStyle], max_width: float) -> list[Any]:
    path = resolve_image(base_dir, data["path"])
    alt = data["alt"]
    if not path.is_file():
        return [Paragraph(f"[FILLME image placeholder: {inline_to_reportlab(alt)}]", styles["Caption"]), Spacer(1, 8)]
    with PILImage.open(path) as im:
        w_px, h_px = im.size
    width = min(max_width, 6.9 * inch)
    height = width * h_px / max(w_px, 1)
    if height > 4.8 * inch:
        height = 4.8 * inch
        width = height * w_px / max(h_px, 1)
    img = Image(str(path), width=width, height=height)
    caption = Paragraph(inline_to_reportlab(alt), styles["Caption"])
    return [KeepTogether([img, Spacer(1, 3), caption]), Spacer(1, 10)]


def resolve_image(base_dir: Path, ref: str) -> Path:
    ref = ref.strip()
    if ref.startswith("http://") or ref.startswith("https://"):
        return Path(ref)
    return (base_dir / ref).resolve()


# ---- DOCX writer ---------------------------------------------------------

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"


def build_docx(md_path: Path, out_path: Path) -> None:
    blocks = parse_markdown(md_path.read_text(encoding="utf-8"))
    media: list[tuple[str, Path, str]] = []
    rels: list[str] = []
    body_parts: list[str] = []
    pic_id = 1

    for block in blocks:
        k, d = block.kind, block.data
        if k == "heading":
            body_parts.append(docx_para(d["text"], style=f"Heading{min(d['level'], 4)}"))
        elif k == "paragraph":
            if is_markdown_visible_caption(d):
                continue
            body_parts.append(docx_para(d))
        elif k == "list":
            for idx, item in enumerate(d["items"], 1):
                prefix = f"{idx}. " if d["ordered"] else "• "
                body_parts.append(docx_para(prefix + item, style="ListParagraph"))
        elif k == "code":
            for line in d["text"].splitlines() or [""]:
                body_parts.append(docx_para(line, style="Code"))
        elif k == "table":
            body_parts.append(docx_table(d))
        elif k == "image":
            path = resolve_image(md_path.parent, d["path"])
            if path.is_file():
                rid = f"rId{len(rels) + 1}"
                ext = path.suffix.lower().lstrip(".") or "png"
                name = f"image{len(media) + 1}.{ext}"
                media.append((name, path, ext))
                rels.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{name}"/>')
                body_parts.append(docx_image(rid, path, pic_id))
                body_parts.append(docx_para(d["alt"], style="Caption"))
                pic_id += 1
            else:
                body_parts.append(docx_para(f"[FILLME image placeholder: {d['alt']}]", style="Caption"))
        elif k == "hr":
            body_parts.append(docx_para(""))

    document_xml = docx_document("\n".join(body_parts))
    rels_xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + "".join(rels) + "</Relationships>"

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types(media))
        z.writestr("_rels/.rels", package_rels())
        z.writestr("word/document.xml", document_xml)
        z.writestr("word/styles.xml", styles_xml())
        z.writestr("word/_rels/document.xml.rels", rels_xml)
        for name, path, _ext in media:
            z.write(path, f"word/media/{name}")


def docx_para(text: str, style: str = "Normal") -> str:
    runs = inline_runs(text)
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>{runs}</w:p>'


def inline_runs(text: str) -> str:
    # Minimal rich-text parser for the report subset: bold (**...** or HTML
    # strong/b), underline (<u>...</u>), inline code (`...`), and line breaks.
    runs: list[str] = []
    buf: list[str] = []
    bold = False
    underline = False
    mono = False
    i = 0

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        chunk = "".join(buf)
        buf = []
        props: list[str] = []
        if bold:
            props.append("<w:b/><w:bCs/>")
        if underline:
            props.append('<w:u w:val="single"/>')
        if mono:
            props.append('<w:rFonts w:ascii="Courier New" w:hAnsi="Courier New"/>')
        rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
        parts = chunk.split("\n")
        for j, part in enumerate(parts):
            if part:
                runs.append(f'<w:r>{rpr}<w:t xml:space="preserve">{escape(part)}</w:t></w:r>')
            if j != len(parts) - 1:
                runs.append("<w:r><w:br/></w:r>")

    while i < len(text):
        if text.startswith("**", i):
            flush()
            bold = not bold
            i += 2
        elif text.startswith("<u>", i):
            flush()
            underline = True
            i += 3
        elif text.startswith("</u>", i):
            flush()
            underline = False
            i += 4
        elif text.startswith("<strong>", i):
            flush()
            bold = True
            i += 8
        elif text.startswith("</strong>", i):
            flush()
            bold = False
            i += 9
        elif text.startswith("<b>", i):
            flush()
            bold = True
            i += 3
        elif text.startswith("</b>", i):
            flush()
            bold = False
            i += 4
        elif text[i] == "`":
            flush()
            mono = not mono
            i += 1
        else:
            buf.append(text[i])
            i += 1
    flush()
    return "".join(runs)


def docx_table(data: dict[str, Any]) -> str:
    rows = [data["header"]] + data["rows"]
    xml = ['<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/><w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/><w:left w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/><w:bottom w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/><w:right w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/><w:insideH w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/><w:insideV w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/></w:tblBorders></w:tblPr>']
    for r_i, row in enumerate(rows):
        xml.append("<w:tr>")
        for cell in row:
            shade = '<w:shd w:fill="E2E8F0"/>' if r_i == 0 else ""
            xml.append(f'<w:tc><w:tcPr>{shade}</w:tcPr>{docx_para(cell)}</w:tc>')
        xml.append("</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


def docx_image(rid: str, path: Path, pic_id: int) -> str:
    with PILImage.open(path) as im:
        w_px, h_px = im.size
    max_w_emu = int(6.2 * 914400)
    width_emu = max_w_emu
    height_emu = int(width_emu * h_px / max(w_px, 1))
    max_h_emu = int(4.6 * 914400)
    if height_emu > max_h_emu:
        height_emu = max_h_emu
        width_emu = int(height_emu * w_px / max(h_px, 1))
    name = escape(path.name)
    return f'''<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{width_emu}" cy="{height_emu}"/><wp:docPr id="{pic_id}" name="{name}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="{pic_id}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{width_emu}" cy="{height_emu}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'''


def docx_document(body: str) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W_NS}" xmlns:r="{R_NS}" xmlns:wp="{WP_NS}" xmlns:a="{A_NS}" xmlns:pic="{PIC_NS}"><w:body>{body}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="900" w:right="900" w:bottom="900" w:left="900" w:header="720" w:footer="720" w:gutter="0"/></w:sectPr></w:body></w:document>'''


def content_types(media: list[tuple[str, Path, str]]) -> str:
    defaults = {
        "rels": "application/vnd.openxmlformats-package.relationships+xml",
        "xml": "application/xml",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
    }
    parts = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">']
    for ext, typ in defaults.items():
        parts.append(f'<Default Extension="{ext}" ContentType="{typ}"/>')
    parts.append('<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>')
    parts.append('<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>')
    parts.append("</Types>")
    return "".join(parts)


def package_rels() -> str:
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'


def styles_xml() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W_NS}">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="{REPORT_FONT_FAMILY}" w:hAnsi="{REPORT_FONT_FAMILY}" w:eastAsia="{REPORT_FONT_FAMILY}" w:cs="{REPORT_FONT_FAMILY}"/><w:sz w:val="20"/></w:rPr></w:rPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="{REPORT_FONT_FAMILY}" w:hAnsi="{REPORT_FONT_FAMILY}" w:eastAsia="{REPORT_FONT_FAMILY}" w:cs="{REPORT_FONT_FAMILY}"/><w:sz w:val="20"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:rPr><w:b/><w:rFonts w:ascii="{REPORT_FONT_FAMILY}" w:hAnsi="{REPORT_FONT_FAMILY}" w:eastAsia="{REPORT_FONT_FAMILY}" w:cs="{REPORT_FONT_FAMILY}"/><w:sz w:val="34"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:rPr><w:b/><w:rFonts w:ascii="{REPORT_FONT_FAMILY}" w:hAnsi="{REPORT_FONT_FAMILY}" w:eastAsia="{REPORT_FONT_FAMILY}" w:cs="{REPORT_FONT_FAMILY}"/><w:sz w:val="28"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:rPr><w:b/><w:rFonts w:ascii="{REPORT_FONT_FAMILY}" w:hAnsi="{REPORT_FONT_FAMILY}" w:eastAsia="{REPORT_FONT_FAMILY}" w:cs="{REPORT_FONT_FAMILY}"/><w:sz w:val="24"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading4"><w:name w:val="heading 4"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:rPr><w:b/><w:rFonts w:ascii="{REPORT_FONT_FAMILY}" w:hAnsi="{REPORT_FONT_FAMILY}" w:eastAsia="{REPORT_FONT_FAMILY}" w:cs="{REPORT_FONT_FAMILY}"/><w:sz w:val="22"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="360"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="Caption"/><w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="center"/></w:pPr><w:rPr><w:i/><w:rFonts w:ascii="{REPORT_FONT_FAMILY}" w:hAnsi="{REPORT_FONT_FAMILY}" w:eastAsia="{REPORT_FONT_FAMILY}" w:cs="{REPORT_FONT_FAMILY}"/><w:color w:val="475569"/><w:sz w:val="18"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/><w:basedOn w:val="Normal"/><w:rPr><w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" w:eastAsia="{REPORT_FONT_FAMILY}" w:cs="Courier New"/><w:sz w:val="18"/></w:rPr></w:style>
</w:styles>'''


def export(md_name: str) -> None:
    md = DOCS / md_name
    if not md.exists():
        raise FileNotFoundError(md)
    stem = md.with_suffix("")
    build_pdf(md, stem.with_suffix(".pdf"))
    build_docx(md, stem.with_suffix(".docx"))
    print(f"Exported {md.name} -> {stem.name}.pdf, {stem.name}.docx")


def main() -> None:
    export("FINAL_REPORT_AI_AIR_DRUM_PAD.md")


if __name__ == "__main__":
    main()
