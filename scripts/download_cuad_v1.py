"""
Download and extract the official CUAD v1 release (full contract text files).

The Atticus Project's CUAD v1 zip contains ~510 full contract .txt files under
full_contract_txt/, matched 1:1 to the contract titles used in the CUAD-QA
SQuAD-format dataset.  This is the correct corpus for evaluation — NOT the
decontextualized excerpt version on HuggingFace (theatticusproject/cuad).

Usage:
    python scripts/download_cuad_v1.py

If the automated download fails, place CUAD_v1.zip manually at
data/cuad_v1_raw.zip and re-run — the script unzips from there.
"""
from __future__ import annotations
import sys
import zipfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logger import logger

CUAD_V1_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip"
DATA_DIR = Path("data/cuad_v1")
ZIP_DEST = Path("data/cuad_v1_raw.zip")


def _progress(count: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    pct = min(count * block_size * 100 // total, 100)
    print(f"\r  Downloading... {pct}%", end="", flush=True)


def download_zip() -> None:
    if ZIP_DEST.exists():
        logger.info(f"Zip already present at {ZIP_DEST}, skipping download.")
        return

    ZIP_DEST.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading CUAD v1 from: {CUAD_V1_URL}")
    try:
        urllib.request.urlretrieve(CUAD_V1_URL, ZIP_DEST, reporthook=_progress)
        print()
        logger.info(f"Saved to {ZIP_DEST}")
    except Exception as exc:
        ZIP_DEST.unlink(missing_ok=True)
        logger.error(f"Download failed: {exc}")
        print("\n--- Manual download instructions ---")
        print("1. Visit https://www.atticusprojectai.org/cuad")
        print("2. Download CUAD_v1.zip")
        print(f"3. Place it at:  {ZIP_DEST.resolve()}")
        print("4. Re-run:  python scripts/download_cuad_v1.py")
        sys.exit(1)


def extract_zip() -> Path:
    """Extract zip and return the path to full_contract_txt/."""
    if not ZIP_DEST.exists():
        logger.error(f"No zip at {ZIP_DEST}. Run download first.")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {ZIP_DEST} -> {DATA_DIR} ...")
    with zipfile.ZipFile(ZIP_DEST, "r") as zf:
        zf.extractall(DATA_DIR)

    txt_dir = _find_txt_dir(DATA_DIR)
    if txt_dir is None:
        logger.error(
            f"Could not locate full_contract_txt/ after extraction in {DATA_DIR}."
        )
        sys.exit(1)

    n_txt = sum(1 for _ in txt_dir.glob("*.txt"))
    logger.info(f"Done. {n_txt} contract .txt files at {txt_dir}")
    return txt_dir


def _find_txt_dir(base: Path) -> Path | None:
    """Recursively locate the full_contract_txt directory."""
    for candidate in sorted(base.rglob("full_contract_txt")):
        if candidate.is_dir() and any(candidate.glob("*.txt")):
            return candidate
    return None


def main() -> None:
    download_zip()
    txt_dir = extract_zip()
    print(f"\nContract text files are at:\n  {txt_dir.resolve()}")
    print("\nNext step — ingest into ChromaDB:")
    print(f"  python scripts/ingest_cuad.py --max 200 --txt-dir \"{txt_dir}\"")


if __name__ == "__main__":
    main()
