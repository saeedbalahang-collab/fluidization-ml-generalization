"""Editable PowerPoint version of Figure 3 (native shapes, text and connectors; every element can be edited in PowerPoint).
Output: results/figures/fig3_workflow_editable.pptx"""
import re, sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fluidml import config as C

prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
sl = prs.slides.add_slide(prs.slide_layouts[6])
SX, SY, X0, Y0 = 1.04, 1.04, 0.35, 0.25          # plot units (12 x 6.4) -> inches
def X(x): return X0 + x * SX
def Y(y): return Y0 + (6.4 - y) * SY               # y axis points up in plot units
def rich(par, text, size, color=RGBColor(0, 0, 0), italic=False):
    """Text with _{sub} and ^{sup} markup."""
    for tok in re.split(r"(_\{[^}]*\}|\^\{[^}]*\})", text):
        if not tok: continue
        r = par.add_run(); sub = tok.startswith("_{"); sup = tok.startswith("^{")
        r.text = tok[2:-1] if (sub or sup) else tok
        f = r.font; f.size = Pt(size); f.name = "Arial"; f.color.rgb = color; f.italic = italic
        if sub: r._r.get_or_add_rPr().set("baseline", "-25000")
        if sup: r._r.get_or_add_rPr().set("baseline", "30000")
def box(x, y, w, h, text, fill="EEF3F8", line="33506B", size=16):
    sh = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(X(x)), Inches(Y(y + h)), Inches(w * SX), Inches(h * SY))
    sh.adjustments[0] = 0.12; sh.fill.solid(); sh.fill.fore_color.rgb = RGBColor.from_string(fill)
    sh.line.color.rgb = RGBColor.from_string(line); sh.line.width = Pt(1.5); sh.shadow.inherit = False
    tf = sh.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for m in ("margin_left", "margin_right"): setattr(tf, m, Inches(0.06))
    for i, line_ in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = PP_ALIGN.CENTER; rich(p, line_, size)
    return sh
def arrow(x1, y1, x2, y2):
    c = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(X(x1)), Inches(Y(y1)), Inches(X(x2)), Inches(Y(y2)))
    c.line.color.rgb = RGBColor.from_string("33506B"); c.line.width = Pt(2)
    ln = c.line._get_or_add_ln(); te = etree.SubElement(ln, qn("a:tailEnd")); te.set("type", "triangle"); te.set("w", "med"); te.set("len", "med")
RF, RE = "FBE9E7", "A33A2A"
box(4.1, 5.5, 3.8, .8, "Experimental dataset: 135 records\n(9 beds × 5 pressures × 3 times)", "FFF5E0", "B07800")
box(0.2, 4.2, 4.4, .9, "Development domain: P = 15–45 m\n(108 records, 9 soil beds)")
box(7.4, 4.2, 4.4, .9, "Withheld domain: P = 55 m\n(27 records; never used for\nfitting, scaling or tuning)", RF, RE)
box(0.2, 3.0, 4.4, .9, "Inputs: X, τ  or  P/CD, CD/d_{50}, τ\nTarget: Y = log_{10}(A_{f}/d_{50}^{2})")
box(0.2, 1.7, 4.4, 1.0, "Grouped splits (all rows of a bed\nkept together): LOBO, LOSO,\nrolling pressure origin")
box(0.2, 0.2, 4.4, 1.2, "Training fold: scaler fitted on\ntraining data only → TPE\nhyper-parameter search\n(inner bed-grouped CV)")
box(4.95, 0.2, 2.1, 1.2, "Final model\nrefit on the\ntraining fold")
box(4.95, 1.7, 2.1, 1.0, "Held-out fold\n(scaler applied,\nnot refitted)")
box(4.95, 3.0, 2.1, .9, "Metrics: log-space,\narea, factor of 2")
box(7.4, 1.7, 4.4, 1.0, "55 m evaluation: model from\ndevelopment data → predict 55 m\n(overall / X in range / above / by soil)", RF, RE, 15)
box(7.4, 0.2, 4.4, 1.2, "Forward-chaining correction test:\ntrain 15–35 m → calibrate 45 m\n→ test 55 m", RF, RE)
for a in [(5.2, 5.5, 2.4, 5.1), (6.8, 5.5, 9.6, 5.1), (2.4, 4.2, 2.4, 3.9), (2.4, 3.0, 2.4, 2.7), (2.4, 1.7, 2.4, 1.4), (4.6, .8, 4.95, .8),
          (6.0, 1.4, 6.0, 1.7), (6.0, 2.7, 6.0, 3.0), (9.6, 4.2, 9.6, 2.7), (9.6, 1.7, 9.6, 1.4)]: arrow(*a)
tb = sl.shapes.add_textbox(Inches(X(0.2)), Inches(Y(-0.05)), Inches(11.6 * SX), Inches(0.4)); p = tb.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
rich(p, "TPE = tree-structured Parzen estimator; LOBO / LOSO = leave-one-bed / leave-one-soil-out", 13, italic=True)
out = C.FIGURES / "fig3_workflow_editable.pptx"; prs.save(out); print("saved", out)
