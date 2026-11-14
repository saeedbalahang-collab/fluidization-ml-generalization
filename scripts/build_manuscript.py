"""Build the clean revised manuscript (.docx) from manuscript/manuscript.md + generated tables."""
import re, subprocess, sys, copy
from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_COLOR_INDEX, WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(__file__).resolve().parents[1]; MS = ROOT / "manuscript"; TB = ROOT / "results" / "tables" / "md"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else MS / "Revised_Manuscript_clean.docx"
src = (MS / "manuscript.md").read_text()
for k in re.findall(r"\{\{(\w+)\}\}", src):
    f = TB / f"{k}.md"
    src = src.replace("{{%s}}" % k, f.read_text() if f.exists() else f"[MISSING {k}]")
(MS / "_build.md").write_text(src)

ref = MS / "_reference.docx"
if not ref.exists():
    ref.write_bytes(subprocess.run(["pandoc", "-o", "/dev/stdout", "--print-default-data-file", "reference.docx"], capture_output=True).stdout)
    d = Document(ref)
    for name, size, bold in (("Normal", 11, False), ("Body Text", 11, False), ("First Paragraph", 11, False), ("Compact", 10, False),
                             ("Image Caption", 9.5, False), ("Table Caption", 9.5, False), ("Title", 16, True), ("Heading 1", 13, True), ("Heading 2", 11.5, True)):
        try:
            st = d.styles[name]
        except KeyError:
            continue
        st.font.name = "Times New Roman"; st.font.size = Pt(size); st.font.bold = bold; st.font.color.rgb = RGBColor(0, 0, 0)
        rpr = st.element.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
        if rf is None: rf = OxmlElement("w:rFonts"); rpr.append(rf)
        for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"): rf.set(qn(a), "Times New Roman")
        for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
            if rf.get(qn(a)) is not None: del rf.attrib[qn(a)]
        if name in ("Body Text", "First Paragraph", "Normal"):
            st.paragraph_format.space_after = Pt(6); st.paragraph_format.line_spacing = 1.3
    d.save(ref)

subprocess.run(["pandoc", str(MS / "_build.md"), "-o", str(out), "--reference-doc", str(ref), "--resource-path", str(MS), "-f", "markdown+tex_math_dollars+pipe_tables+superscript+subscript+table_captions"], check=True)

d = Document(out)
sec = d.sections[0]; sec.page_width, sec.page_height = Cm(21), Cm(29.7)
for a in ("left_margin", "right_margin"): setattr(sec, a, Cm(2.3))
sec.top_margin = sec.bottom_margin = Cm(2.3)
ln = OxmlElement("w:lnNumType"); ln.set(qn("w:countBy"), "1"); ln.set(qn("w:restart"), "continuous"); sec._sectPr.append(ln)   # line numbers
# page numbers in footer
fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = fp.add_run()
for t, txt in (("begin", None), (None, "PAGE"), ("end", None)):
    if t: e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), t)
    else: e = OxmlElement("w:instrText"); e.text = txt
    r._r.append(e)
# three-line tables
def border(cell_or_tbl_pr, **kw):
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom", "insideH"):
        e = OxmlElement(f"w:{edge}"); v = kw.get(edge, "nil"); e.set(qn("w:val"), v); e.set(qn("w:sz"), "6"); e.set(qn("w:color"), "000000"); b.append(e)
    cell_or_tbl_pr.append(b)
for t in d.tables:
    pr = t._tbl.tblPr
    for old in pr.findall(qn("w:tblBorders")): pr.remove(old)
    border(pr, top="single", bottom="single", insideH="nil")
    for i, row in enumerate(t.rows):
        for c in row.cells:
            for p in c.paragraphs:
                p.paragraph_format.space_after = Pt(1); p.paragraph_format.line_spacing = 1.0
                for rn in p.runs: rn.font.size = Pt(8.5); rn.font.name = "Times New Roman"; rn.font.bold = (i == 0) or rn.font.bold
            if i == 0:
                tcPr = c._tc.get_or_add_tcPr(); bd = OxmlElement("w:tcBorders"); e = OxmlElement("w:bottom"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "6"); bd.append(e); tcPr.append(bd)
# column widths proportional to content length
for t in d.tables:
    n = len(t.columns); tot = Cm(16.4)
    L = [max(len(c.text) for c in col.cells) if True else 1 for col in t.columns]
    L = [min(max(l, 8), 60) for l in L]; w = [int(tot * l / sum(L)) for l in L]
    t.autofit = False
    for row in t.rows:
        for i, c in enumerate(row.cells): c.width = w[i]
    for i, gc in enumerate(t._tbl.tblGrid.findall(qn("w:gridCol"))): gc.set(qn("w:w"), str(int(w[i] / 635)))
# highlight author placeholders
pat = re.compile(r"\[AUTHORS[^\]]*\]")
def hl(par):
    for run in list(par.runs):
        m = pat.search(run.text)
        if not m: continue
        txt = run.text; before, mid, after = txt[:m.start()], m.group(0), txt[m.end():]
        run.text = before
        r2 = copy.deepcopy(run._r); run._r.addnext(r2)
        from docx.text.run import Run
        R2 = Run(r2, par); R2.text = mid; R2.font.highlight_color = WD_COLOR_INDEX.YELLOW
        if after:
            r3 = copy.deepcopy(run._r); r2.addnext(r3); R3 = Run(r3, par); R3.text = after
            hl(par); return
for p in d.paragraphs: hl(p)
for t in d.tables:
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs: hl(p)
for p_ in d.paragraphs:
    if p_.style.name.startswith(('Heading','Title')):
        for r_ in p_.runs: r_.font.color.rgb = RGBColor(0, 0, 0); r_.font.name = 'Times New Roman'
d.save(out); print("saved", out)
