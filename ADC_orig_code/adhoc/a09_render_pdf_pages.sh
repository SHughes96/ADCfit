#!/bin/bash
# Render each page of the colleague PDF to PNG for visual checking (macOS QuickLook + pypdf).
PDF=/Users/slh/Documents/Projects/michael_reduction/lya_analysis/LLAMAS_Lya_summary_2026-10-16.pdf
OUTD=${1:-/tmp/lya_pdf_pages}; mkdir -p "$OUTD"
conda run -n llamas_data_reduction_clean python -c "
from pypdf import PdfReader, PdfWriter
r = PdfReader('$PDF')
for i, p in enumerate(r.pages):
    w = PdfWriter(); w.add_page(p); w.write('$OUTD/page%d.pdf' % (i + 1))
print(len(r.pages), 'pages')"
for f in "$OUTD"/page*.pdf; do qlmanage -t -s 1400 -o "$OUTD" "$f" >/dev/null 2>&1; done
ls "$OUTD"
