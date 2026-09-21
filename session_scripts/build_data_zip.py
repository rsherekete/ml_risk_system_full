"""Zip the antifraud data files (stored, not compressed: parquet is already
compressed) with their in-repo paths, so unzipping over a clone puts them
where the app reads them."""
import sys
import time
import zipfile
from pathlib import Path

NOTEBOOK = Path(r"C:\Users\RoyVivasi\Documents\notebook")
sys.path.insert(0, str(NOTEBOOK / "tools"))
from github_data_assets import MANIFEST, _source  # noqa: E402

out = NOTEBOOK / "docs" / "antifraud_data_2026-09-14.zip"
tmp = out.with_suffix(".zip.part")
t0 = time.time()
readme = (
    "Antifraud data snapshot 2026-09-14 for the notebook app.\n\n"
    "Unzip this archive over the folder that contains webapp/ (the repository root).\n"
    "Every file lands at the path the app reads:\n\n"
    + "".join(f"  {dest}\n" for dest in MANIFEST.values())
    + "\nThen restart the app. webapp/model_service.py reads these from webapp/artifacts/ad\n"
      "when NOTEBOOK_DATA_DIR is unset. The same files are also attached to the GitHub\n"
      "release data-2026-09-14 (python tools/github_data_assets.py fetch).\n")
with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
    zf.writestr("README_DATA.txt", readme)
    for name, dest in MANIFEST.items():
        src = _source(name)
        if src is None:
            print(f"{name}: no local copy, skipped")
            continue
        print(f"{name}: {src.stat().st_size / 1e6:,.1f} MB from {src}")
        zf.write(src, dest)
tmp.replace(out)
print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB) in {time.time() - t0:.0f}s")
