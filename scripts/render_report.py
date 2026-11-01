#!/usr/bin/env python3
"""Render the compact Markdown report using installed, portable PDF tools."""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def font_paths():
    options = [
        (Path('/System/Library/Fonts/Supplemental/Arial.ttf'), Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf')),
        (Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')),
    ]
    for regular, bold in options:
        if regular.exists() and bold.exists():
            return regular, bold
    raise FileNotFoundError('Install Arial or DejaVu Sans (including bold) for Cyrillic report text.')


def inline(text):
    text = html.escape(text, quote=False)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<a href="\2" color="#225588">\1</a>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`([^`]+)`', r'<font name="ReportFont">\1</font>', text)
    return text


def render(source, destination):
    source, destination = Path(source), Path(destination)
    regular, bold = font_paths()
    pdfmetrics.registerFont(TTFont('ReportFont', str(regular)))
    pdfmetrics.registerFont(TTFont('ReportBold', str(bold)))
    pdfmetrics.registerFontFamily('ReportFont', normal='ReportFont', bold='ReportBold', italic='ReportFont', boldItalic='ReportBold')
    body = ParagraphStyle('body', fontName='ReportFont', fontSize=10, leading=12.4, spaceAfter=5, textColor=colors.HexColor('#18212c'), splitLongWords=True)
    h1 = ParagraphStyle('h1', parent=body, fontName='ReportBold', fontSize=18, leading=21, spaceAfter=10)
    h2 = ParagraphStyle('h2', parent=body, fontName='ReportBold', fontSize=12.5, leading=15, spaceBefore=7, spaceAfter=6, keepWithNext=True)
    h3 = ParagraphStyle('h3', parent=body, fontName='ReportBold', fontSize=10.8, leading=13, spaceBefore=5, spaceAfter=4, keepWithNext=True)
    cell = ParagraphStyle('cell', parent=body, fontSize=8.4, leading=10.4, spaceAfter=0)
    bullet = ParagraphStyle('bullet', parent=body, leftIndent=11, firstLineIndent=-8, spaceAfter=3)
    reference = ParagraphStyle('reference', parent=body, fontSize=9, leading=11.2, spaceAfter=7)
    width = A4[0] - 82
    doc = SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=41, leftMargin=41, topMargin=35, bottomMargin=35,
                            title='Temporal TV Programme Recommendation — Individual LLM4Rec/FKN', author='Vladislav Neklyudov')
    story, lines, i = [], source.read_text().splitlines(), 0
    references = False
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == '<!-- pagebreak -->':
            story.append(PageBreak())
            i += 1
            continue
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            title = line[level:].strip()
            references = references or title == 'References'
            story.append(Paragraph(inline(title), {1: h1, 2: h2}.get(level, h3)))
            i += 1
            continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in cells):
                    rows.append(cells)
                i += 1
            columns = len(rows[0])
            if columns == 6:
                fractions = [0.16, 0.13, 0.14, 0.16, 0.24, 0.17]
            elif columns == 4:
                fractions = [0.23, 0.20, 0.32, 0.25]
            elif columns == 3:
                fractions = [0.18, 0.66, 0.16]
            elif columns == 5:
                fractions = [0.12, 0.43, 0.17, 0.14, 0.14]
            else:
                fractions = [1 / columns] * columns
            table = Table([[Paragraph(inline(c), cell) for c in row] for row in rows],
                          colWidths=[f * width for f in fractions], hAlign='LEFT', repeatRows=1)
            table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8edf3')),
                                       ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LINEBELOW', (0, 0), (-1, 0), 0.6, colors.HexColor('#6b7a90')),
                                       ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f6f8fa')]),
                                       ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                                       ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
            story += [table, Spacer(1, 6)]
            continue
        image_match = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line)
        if image_match:
            image_path = source.parent / image_match.group(2)
            if image_path.exists():
                picture = Image(str(image_path))
                scale = min(width / picture.imageWidth, 280 / picture.imageHeight)
                picture.drawWidth, picture.drawHeight = picture.imageWidth * scale, picture.imageHeight * scale
                story += [picture, Spacer(1, 7), Paragraph(inline(image_match.group(1)), cell)]
            else:
                placeholder = Table([[Paragraph('<b>REQUIRED GENUINE SCREENSHOT PENDING</b><br/><br/>'
                                               'Insert the designated COMPLETE Kaggle submission page showing its private score. '
                                               'No score image has been fabricated.<br/><br/>' + inline(str(image_path)), body)]],
                                    colWidths=[width], rowHeights=[190])
                placeholder.setStyle(TableStyle([('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#9a4b32')),
                                                 ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fff5ef')),
                                                 ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('LEFTPADDING', (0, 0), (-1, -1), 18),
                                                 ('RIGHTPADDING', (0, 0), (-1, -1), 18)]))
                story.append(placeholder)
            i += 1
            continue
        is_bullet = line.startswith('- ')
        chunks = [line[2:] if is_bullet else line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(('#', '|', '- ', '<!--', '![')):
            chunks.append(lines[i].strip())
            i += 1
        text = ' '.join(chunks)
        story.append(Paragraph(('• ' if is_bullet else '') + inline(text), reference if references else bullet if is_bullet else body))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont('ReportFont', 8)
        canvas.setFillColor(colors.HexColor('#687789'))
        canvas.drawString(41, 20, 'Individual LLM4Rec/FKN · 1 November 2026')
        canvas.drawRightString(A4[0] - 41, 20, str(document.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return str(destination)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source', nargs='?', default='report/report.md')
    p.add_argument('destination', nargs='?', default='report/report.pdf')
    a = p.parse_args()
    print(render(a.source, a.destination))
