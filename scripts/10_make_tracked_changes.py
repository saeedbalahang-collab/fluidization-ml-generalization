"""Create a tracked-changes copy of the revised manuscript relative to the originally submitted .docx (LibreOffice compare).
usage: python scripts/10_make_tracked_changes.py ORIGINAL.docx REVISED_CLEAN.docx OUT.docx
Requires LibreOffice with python-uno. (Alternative: Word > Review > Compare.)"""
import subprocess, sys, time, os
from pathlib import Path
import uno
from com.sun.star.beans import PropertyValue

orig, rev, out = (Path(a).resolve() for a in sys.argv[1:4])
port = 2002
proc = subprocess.Popen(["soffice", "--headless", "--invisible", "--norestore", f"--accept=socket,host=localhost,port={port};urp;"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
pv = lambda n, v: PropertyValue(Name=n, Value=v)
try:
    ctx = None
    for _ in range(60):
        try:
            local = uno.getComponentContext()
            resolver = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)
            ctx = resolver.resolve(f"uno:socket,host=localhost,port={port};urp;StarOffice.ComponentContext"); break
        except Exception: time.sleep(1)
    smgr = ctx.ServiceManager; desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
    doc = desktop.loadComponentFromURL(rev.as_uri(), "_blank", 0, (pv("Hidden", True),))
    disp = smgr.createInstanceWithContext("com.sun.star.frame.DispatchHelper", ctx)
    disp.executeDispatch(doc.CurrentController.Frame, ".uno:CompareDocuments", "", 0, (pv("URL", orig.as_uri()), pv("NoAcceptDialog", True)))
    doc.storeToURL(out.as_uri(), (pv("FilterName", "MS Word 2007 XML"),)); doc.close(True)
    print("saved", out)
finally:
    try: desktop.terminate()
    except Exception: pass
    proc.terminate()
