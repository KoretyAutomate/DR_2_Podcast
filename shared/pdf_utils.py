"""
shared/pdf_utils.py — PDF generation utilities with markdown-aware rendering.
"""

import re
from pathlib import Path
from fpdf import FPDF


class SciencePDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'DGX Spark Research Intelligence Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


def _safe_latin1(text: str) -> str:
    """Encode text to latin-1, dropping unsupported characters."""
    return text.encode('latin-1', 'ignore').decode('latin-1')


def _render_inline_bold(pdf, text: str, font_size: int = 11):
    """Render text with **bold** segments as inline bold within a line."""
    parts = re.split(r'(\*\*.*?\*\*)', text)
    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            pdf.set_font("Helvetica", "B", font_size)
            pdf.write(6, _safe_latin1(part[2:-2]))
        else:
            pdf.set_font("Helvetica", "", font_size)
            pdf.write(6, _safe_latin1(part))


def _parse_table(lines: list) -> list:
    """Parse markdown table lines into a list of rows (each row is a list of cells).
    Skips the separator row (| --- | --- |)."""
    rows = []
    for line in lines:
        line = line.strip()
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.split('|')]
        # Remove empty first/last from leading/trailing pipes
        if cells and cells[0] == '':
            cells = cells[1:]
        if cells and cells[-1] == '':
            cells = cells[:-1]
        # Skip separator rows
        if all(re.match(r'^-+:?-*$|^:?-+:?$', c) for c in cells):
            continue
        if cells:
            rows.append(cells)
    return rows


def _render_table(pdf, rows: list):
    """Render a parsed markdown table as a PDF table with borders."""
    if not rows:
        return

    page_width = pdf.w - pdf.l_margin - pdf.r_margin
    num_cols = max(len(row) for row in rows)
    if num_cols == 0:
        return

    # Calculate column widths based on content
    col_max_chars = [0] * num_cols
    for row in rows:
        for i, cell in enumerate(row):
            if i < num_cols:
                col_max_chars[i] = max(col_max_chars[i], len(cell))

    total_chars = max(sum(col_max_chars), 1)
    col_widths = [(c / total_chars) * page_width for c in col_max_chars]

    # Enforce minimum width
    min_width = 15
    for i in range(num_cols):
        if col_widths[i] < min_width:
            col_widths[i] = min_width

    # Normalize to fit page
    total_w = sum(col_widths)
    if total_w > page_width:
        scale = page_width / total_w
        col_widths = [w * scale for w in col_widths]

    row_height = 7
    font_size = 8

    for row_idx, row in enumerate(rows):
        # Check if we need a new page
        if pdf.get_y() + row_height > pdf.h - pdf.b_margin - 15:
            pdf.add_page()

        x_start = pdf.get_x()
        y_start = pdf.get_y()

        # Calculate row height based on content wrapping
        max_lines = 1
        for i in range(min(len(row), num_cols)):
            cell_text = _safe_latin1(re.sub(r'\*\*', '', row[i]))
            char_width = col_widths[i] / (font_size * 0.5) if font_size > 0 else 20
            lines_needed = max(1, len(cell_text) // max(int(char_width), 1) + 1)
            max_lines = max(max_lines, lines_needed)

        actual_row_height = row_height * max_lines

        for i in range(num_cols):
            cell_text = row[i] if i < len(row) else ""
            cell_text = _safe_latin1(re.sub(r'\*\*', '', cell_text))

            # Header row gets bold
            if row_idx == 0:
                pdf.set_font("Helvetica", "B", font_size)
            else:
                pdf.set_font("Helvetica", "", font_size)

            pdf.set_xy(x_start + sum(col_widths[:i]), y_start)

            # Draw cell border and text
            pdf.cell(col_widths[i], actual_row_height, cell_text[:50], border=1, ln=0, align='L')

        pdf.ln(actual_row_height)

    pdf.ln(4)


def _parse_markdown_blocks(content: str) -> list:
    """Parse markdown content into structured blocks for rendering.

    Returns a list of dicts with 'type' and 'content' keys:
    - heading: {type, level, content}
    - paragraph: {type, content}
    - bullet: {type, content}
    - numbered: {type, number, content}
    - table: {type, rows}  (rows = list of cell lists)
    """
    blocks = []
    lines = content.split('\n')
    i = 0
    table_buffer = []

    def flush_table():
        nonlocal table_buffer
        if table_buffer:
            rows = _parse_table(table_buffer)
            if rows:
                blocks.append({'type': 'table', 'rows': rows})
            table_buffer = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Detect table lines
        if stripped.startswith('|') and '|' in stripped[1:]:
            table_buffer.append(stripped)
            i += 1
            continue
        else:
            flush_table()

        # Empty line
        if not stripped:
            i += 1
            continue

        # Headings
        heading_match = re.match(r'^(#{1,4})\s+(.*)', stripped)
        if heading_match:
            level = len(heading_match.group(1))
            blocks.append({'type': 'heading', 'level': level, 'content': heading_match.group(2)})
            i += 1
            continue

        # Bullet lists
        bullet_match = re.match(r'^[-*]\s+(.*)', stripped)
        if bullet_match:
            blocks.append({'type': 'bullet', 'content': bullet_match.group(1)})
            i += 1
            continue

        # Numbered lists
        num_match = re.match(r'^(\d+)\.\s+(.*)', stripped)
        if num_match:
            blocks.append({'type': 'numbered', 'number': num_match.group(1), 'content': num_match.group(2)})
            i += 1
            continue

        # Regular paragraph — collect consecutive non-empty lines
        para_lines = [stripped]
        i += 1
        while i < len(lines):
            next_stripped = lines[i].strip()
            if not next_stripped:
                break
            if next_stripped.startswith('#') or next_stripped.startswith('|') or \
               re.match(r'^[-*]\s+', next_stripped) or re.match(r'^\d+\.\s+', next_stripped):
                break
            para_lines.append(next_stripped)
            i += 1
        blocks.append({'type': 'paragraph', 'content': ' '.join(para_lines)})

    flush_table()
    return blocks


def create_pdf(title: str, content: str, filename: str, output_dir: Path, language: str = "en") -> Path:
    """Create PDF with markdown-aware rendering.

    Parses markdown headings, bold, tables, and bullet lists into
    properly formatted PDF elements.
    """
    pdf = SciencePDF()
    pdf.add_page()

    clean_content = re.sub(r'<think>.*?</think>', '', str(content), flags=re.DOTALL)

    if language == 'ja':
        print("Warning: Japanese characters may not display correctly in PDF. Consider upgrading to fpdf2.")

    clean_title = _safe_latin1(title)

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, clean_title, 0, 1, 'L')
    pdf.ln(3)

    # Parse and render markdown blocks
    blocks = _parse_markdown_blocks(clean_content)

    for block in blocks:
        # Check page break
        if pdf.get_y() > pdf.h - pdf.b_margin - 20:
            pdf.add_page()

        if block['type'] == 'heading':
            level = block['level']
            text = _safe_latin1(re.sub(r'\*\*', '', block['content']))
            if level == 1:
                pdf.set_font("Helvetica", "B", 16)
                pdf.ln(6)
                pdf.cell(0, 10, text, 0, 1, 'L')
                # Underline
                pdf.set_draw_color(100, 100, 100)
                y = pdf.get_y()
                pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
                pdf.ln(3)
            elif level == 2:
                pdf.set_font("Helvetica", "B", 14)
                pdf.ln(5)
                pdf.cell(0, 8, text, 0, 1, 'L')
                # Light underline
                pdf.set_draw_color(180, 180, 180)
                y = pdf.get_y()
                pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
                pdf.ln(2)
            elif level == 3:
                pdf.set_font("Helvetica", "B", 12)
                pdf.ln(4)
                pdf.cell(0, 7, text, 0, 1, 'L')
                pdf.ln(1)
            else:
                pdf.set_font("Helvetica", "B", 11)
                pdf.ln(3)
                pdf.cell(0, 7, text, 0, 1, 'L')
                pdf.ln(1)

        elif block['type'] == 'table':
            _render_table(pdf, block['rows'])

        elif block['type'] == 'bullet':
            pdf.set_font("Helvetica", "", 11)
            x = pdf.get_x()
            pdf.cell(8, 6, chr(149), 0, 0)  # bullet character
            _render_inline_bold(pdf, block['content'], 11)
            pdf.ln(6)

        elif block['type'] == 'numbered':
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(8, 6, f"{block['number']}.", 0, 0)
            _render_inline_bold(pdf, block['content'], 11)
            pdf.ln(6)

        elif block['type'] == 'paragraph':
            _render_inline_bold(pdf, block['content'], 11)
            pdf.ln(8)

    file_path = output_dir / filename
    pdf.output(str(file_path))
    print(f"PDF Generated: {file_path}")
    return file_path


def save_markdown(label: str, source, filename: str, output_dir: Path):
    """Save a task output or raw string as a markdown file."""
    try:
        if isinstance(source, str):
            content = source
        elif hasattr(source, 'output') and source.output and hasattr(source.output, 'raw'):
            content = source.output.raw
        else:
            content = None
        if content and content.strip():
            outfile = output_dir / filename
            with open(outfile, 'w') as f:
                f.write(content)
            print(f"  Saved {filename} ({len(content)} chars)")
        else:
            print(f"  Skipping {filename}: no output available")
    except Exception as e:
        print(f"  Warning: Could not save {filename}: {e}")


def save_pdf_safe(title: str, source, filename: str, output_dir: Path, language: str = "en"):
    """Safely attempt to create a PDF from a task output."""
    try:
        if isinstance(source, str):
            content = source
        elif hasattr(source, 'output') and source.output and hasattr(source.output, 'raw'):
            content = source.output.raw
        else:
            print(f"  Skipping {filename}: no output available")
            return
        create_pdf(title, content, filename, output_dir, language)
    except Exception as e:
        print(f"  Warning: Failed to create {filename}: {e}")
