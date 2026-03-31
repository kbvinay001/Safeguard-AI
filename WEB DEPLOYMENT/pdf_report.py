"""
pdf_report.py — SafeGuard AI PDF Safety Report Generator
=========================================================
Generates a professionally formatted A4 landscape PDF report
summarising the results of a safety detection session.

The report includes:
  - Session summary (source, frames analysed, avg FPS, compliance score)
  - Compliance assessment with a colour-coded EXCELLENT / GOOD / CRITICAL band
  - Alert breakdown (PPE violations vs tool abandonments)
  - Alert log table (up to 50 rows; overflow noted)
  - Actionable safety recommendations based on the alert types found
  - Model performance reference table (mAP, precision, recall for all 3 models)

Usage:
    from pdf_report import generate_pdf
    pdf_bytes = generate_pdf(stats_dict, alerts_list, session_id)

The returned bytes object can be written to a .pdf file or served as
a Streamlit download button response.

Requires: fpdf2  (pip install fpdf2)
"""

from fpdf import FPDF
import datetime
import io


def _sanitise_for_latin1(text: str) -> str:
    """
    Replace common Unicode characters that the Helvetica font (Latin-1) cannot render.

    fpdf2 with Helvetica is limited to Latin-1 (ISO-8859-1). Without this
    sanitisation, em-dashes, curly quotes, and non-breaking spaces cause
    rendering errors or silent truncation.

    Args:
        text (str): Raw string that may contain Unicode characters

    Returns:
        str: String safe for Latin-1 encoding
    """
    replacements = {
        "\u2014": "-",   # em dash
        "\u2013": "-",   # en dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2022": "*",   # bullet point
        "\xa0":   " ",   # non-breaking space
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text.encode("latin-1", errors="replace").decode("latin-1")


class SafetyReport(FPDF):
    """
    Custom FPDF subclass that adds a branded header, page footer,
    section title bar, key-value row, and alert table to every page.
    """

    def header(self):
        """Render the dark branded header bar at the top of each page."""
        self.set_font("Helvetica", "B", 14)
        self.set_fill_color(13, 17, 23)    # Near-black background
        self.set_text_color(255, 255, 255)
        self.cell(
            0, 12,
            "SafeGuard AI  |  Industrial Safety Detection Report",
            align="C", fill=True,
            new_x="LMARGIN", new_y="NEXT"
        )
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def footer(self):
        """Render the page number and generation timestamp at the bottom of each page."""
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.cell(
            0, 8,
            f"SafeGuard AI  |  Generated {timestamp}  |  Page {self.page_no()}",
            align="C"
        )

    def section_title(self, title: str):
        """Render a dark section header bar with blue label text."""
        self.set_font("Helvetica", "B", 11)
        self.set_fill_color(22, 27, 34)    # Dark blue-grey
        self.set_text_color(96, 165, 250)  # Blue label text
        self.cell(0, 8, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def kv_row(self, key: str, value, shade: bool = False):
        """
        Render one key-value pair as a two-column row.

        Args:
            key   (str):  Label in the left column (bold)
            value       :  Value in the right column (regular weight)
            shade (bool): True to use a light grey background (alternating rows)
        """
        self.set_font("Helvetica", "B", 9)
        bg = (240, 244, 248) if shade else (255, 255, 255)
        self.set_fill_color(*bg)
        self.cell(65, 7, _sanitise_for_latin1(f"  {key}"), fill=True)
        self.set_font("Helvetica", "", 9)
        self.cell(0, 7, _sanitise_for_latin1(str(value)), fill=True, new_x="LMARGIN", new_y="NEXT")

    def alert_table(self, alerts: list):
        """
        Render a tabular alert log showing up to 50 rows.

        Columns: # | Time | Type | Tool | Timer (s) | Missing PPE

        Args:
            alerts (list[dict]): Alert records from the database or session stats
        """
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(22, 27, 34)
        self.set_text_color(255, 255, 255)

        column_labels  = ["#",  "Time", "Type", "Tool / Class", "Timer (s)", "Missing PPE"]
        column_widths  = [10,   32,     38,     38,             22,          50]

        for width, label in zip(column_widths, column_labels):
            self.cell(width, 7, label, fill=True)
        self.ln()

        self.set_text_color(0, 0, 0)

        for index, alert in enumerate(alerts[:50]):
            self.set_font("Helvetica", "", 7.5)
            shaded = index % 2 == 0
            bg = (245, 248, 252) if shaded else (255, 255, 255)
            self.set_fill_color(*bg)

            self.cell(10,  6, str(index + 1),                              fill=shaded)
            self.cell(32,  6, str(alert.get("ts",        ""))[:19],        fill=shaded)
            self.cell(38,  6, str(alert.get("type",      ""))[:20],        fill=shaded)
            self.cell(38,  6, str(alert.get("tool",  "N/A"))[:20],         fill=shaded)
            self.cell(22,  6, f"{float(alert.get('timer_s', 0)):.1f}",     fill=shaded)
            self.cell(50,  6, str(alert.get("missing_ppe", ""))[:28],      fill=shaded)
            self.ln()

        if len(alerts) > 50:
            self.set_font("Helvetica", "I", 8)
            self.cell(
                0, 6,
                f"  ... and {len(alerts) - 50} more alerts — see the CSV export for the full log",
                new_x="LMARGIN", new_y="NEXT"
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_pdf(stats: dict, alerts: list, session_id: str) -> bytes:
    """
    Generate a complete safety report PDF and return it as bytes.

    Args:
        stats      (dict): Session statistics including frame count, alert count,
                           compliance score, avg FPS, and source path.
        alerts     (list): List of alert dicts fetched from the database.
        session_id (str):  Human-readable session identifier for the report header.

    Returns:
        bytes: Raw PDF file content ready to write to disk or serve via HTTP.
    """
    pdf = SafetyReport(orientation="L", format="A4")
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()

    # Pull summary values from the stats dict (with safe defaults)
    total_frames  = stats.get("total_frames", 0)
    total_alerts  = stats.get("total_alerts", 0)
    compliance    = stats.get("compliance",   100.0)
    avg_fps       = stats.get("avg_fps",      0.0)
    source        = stats.get("source",       "N/A")
    generated_at  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Section 1 — Report summary key-value block
    pdf.section_title("REPORT SUMMARY")
    pdf.kv_row("Report Generated",   generated_at,                              shade=False)
    pdf.kv_row("Session ID",         session_id,                                shade=True)
    pdf.kv_row("Source",             source,                                    shade=False)
    pdf.kv_row("Frames Analysed",    total_frames,                              shade=True)
    pdf.kv_row("Avg Processing FPS", f"{avg_fps:.1f} FPS",                     shade=False)
    pdf.kv_row("Total Alerts",       total_alerts,                              shade=True)
    pdf.kv_row("Safety Compliance",  f"{compliance:.1f}%",                     shade=False)
    pdf.kv_row("Detection Models",   "3x YOLOv11n — Human / Tool / PPE",       shade=True)
    pdf.kv_row("Alert Logic",        "Temporal FSM — T1=25s Warning, T2=35s Alert", shade=False)
    pdf.ln(4)

    # Section 2 — Compliance band (colour-coded EXCELLENT / GOOD / CRITICAL)
    pdf.section_title("COMPLIANCE ASSESSMENT")
    if   compliance >= 90:
        band, r, g, b = "EXCELLENT",                        21,  128,  61
    elif compliance >= 75:
        band, r, g, b = "GOOD",                            161,   98,   7
    elif compliance >= 60:
        band, r, g, b = "NEEDS IMPROVEMENT",               194,   65,  12
    else:
        band, r, g, b = "CRITICAL — IMMEDIATE ACTION REQUIRED", 185, 28, 28

    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(r, g, b)
    pdf.cell(
        0, 10,
        _sanitise_for_latin1(f"  Overall Safety Status: {band}  ({compliance:.1f}%)"),
        new_x="LMARGIN", new_y="NEXT"
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    # Section 3 — Alert type breakdown
    pdf.section_title("ALERT BREAKDOWN")
    ppe_violations     = sum(1 for a in alerts if a.get("type") == "PPE_VIOLATION")
    tool_abandonments  = sum(1 for a in alerts if a.get("type") == "TOOL_UNATTENDED")
    pdf.kv_row("PPE Violations",    ppe_violations,    shade=False)
    pdf.kv_row("Tool Abandonments", tool_abandonments, shade=True)
    pdf.ln(4)

    # Section 4 — Alert log table (up to 50 rows)
    if alerts:
        pdf.section_title(f"ALERT LOG  (showing up to 50 of {len(alerts)})")
        pdf.alert_table(alerts)

    # Section 5 — Recommendations (second page)
    pdf.add_page()
    pdf.section_title("SAFETY RECOMMENDATIONS")

    recommendations = []
    if ppe_violations > 0:
        recommendations.append("Enforce mandatory PPE checks at all site entry points.")
        recommendations.append("Schedule a PPE compliance training refresher for all workers this week.")
    if tool_abandonments > 0:
        recommendations.append("Implement a tool sign-in / sign-out protocol to reduce abandonment incidents.")
        recommendations.append("Mark designated tool storage zones clearly on the factory floor.")
    if not recommendations:
        recommendations.append("No critical violations detected this session. Maintain current safety protocols.")
        recommendations.append("Schedule the next routine safety audit within 30 days.")

    for index, recommendation in enumerate(recommendations, start=1):
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(
            0, 8,
            _sanitise_for_latin1(f"  {index}. {recommendation}"),
            new_x="LMARGIN", new_y="NEXT"
        )
    pdf.ln(4)

    # Section 6 — Model performance reference
    pdf.section_title("MODEL PERFORMANCE REFERENCE (from training)")
    pdf.kv_row("Human Detection", "mAP@50: 99.44%  |  Precision: 99.20%  |  Recall: 99.26%", shade=False)
    pdf.kv_row("Tool Detection",  "mAP@50: 67.87%  |  Precision: 72.10%  |  Recall: 65.90%", shade=True)
    pdf.kv_row("PPE Detection",   "mAP@50: 79.90%  |  Precision: 81.94%  |  Recall: 71.48%", shade=False)

    return bytes(pdf.output())
