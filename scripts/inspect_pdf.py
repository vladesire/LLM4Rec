#!/usr/bin/env python3
"""Inspect page count, text bounds, glyphs and screenshot presence after rebuilding."""
from pathlib import Path
import json

import pymupdf

from data_pipeline import dump, sha256


def main():
    path = Path('report/report.pdf')
    pdf = pymupdf.open(path)
    reference_page = None
    overflow, replacement_glyphs = [], 0
    pages = []
    preview_dir = Path('/tmp/llm4rec-report-preview')
    preview_dir.mkdir(exist_ok=True)
    for i, page in enumerate(pdf):
        text = page.get_text()
        if '\nReferences\n' in text:
            reference_page = i + 1
        replacement_glyphs += text.count('\ufffd')
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    x0, y0, x1, y1 = span['bbox']
                    if x0 < 0 or y0 < 0 or x1 > page.rect.width + .5 or y1 > page.rect.height + .5:
                        overflow.append({'page': i + 1, 'text': span['text'], 'bbox': span['bbox']})
        page.get_pixmap(matrix=pymupdf.Matrix(1.2, 1.2)).save(preview_dir / f'page-{i+1}.png')
        pages.append({'page': i + 1, 'text_characters': len(text), 'embedded_images': len(page.get_images())})
    content_pages = reference_page - 1 if reference_page else len(pdf)
    details = json.loads(Path('report/student_details.json').read_text())
    screenshot_present = (Path('report') / details['score_screenshot']).exists()
    checks = {'pdf_sha256': sha256(path), 'markdown_sha256': sha256('report/report.md'),
              'total_pages': len(pdf), 'content_pages_excluding_references': content_pages,
              'at_most_five_content_pages': content_pages <= 5, 'reference_starts_on_page': reference_page,
              'out_of_page_text_spans': overflow, 'replacement_glyph_count': replacement_glyphs,
              'genuine_score_screenshot_present': screenshot_present,
              'visual_review_note': 'This script checks geometry, extracted text and page count only. Actual agent visual review is recorded separately in pdf_visual_review.json, tied to its PDF checksum. A genuine screenshot requires a new visual review.',
              'student_must_reinspect_after_final_edits': True, 'preview_directory': str(preview_dir), 'pages': pages}
    dump('results/pdf_inspection.json', checks)
    assert content_pages <= 5, 'Report exceeds five content pages; shorten personal text or adjust layout.'
    assert not overflow and replacement_glyphs == 0
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
