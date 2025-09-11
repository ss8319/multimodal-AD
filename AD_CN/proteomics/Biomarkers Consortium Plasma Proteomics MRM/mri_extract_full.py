import os
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable
from multiprocessing import Pool, cpu_count

# Globals used by worker processes (set by _init_pool)
_ZIP_PATH = None
_OUT_ROOT: Path | None = None
_MAX_LEN: int | None = None
_OVERWRITE: bool | None = None
_ZF: zipfile.ZipFile | None = None


def _init_pool(zp: str, out_dir: str, max_len: int, ow: bool):
    """Initializer for worker processes (Windows-friendly)."""
    global _ZIP_PATH, _OUT_ROOT, _MAX_LEN, _OVERWRITE, _ZF
    _ZIP_PATH = zp
    _OUT_ROOT = Path(out_dir)
    _MAX_LEN = max_len
    _OVERWRITE = ow
    _ZF = zipfile.ZipFile(_ZIP_PATH, 'r')


def _extract_one(name: str) -> tuple[int, int, int]:
    """Extract a single file; returns (written, skipped, errors)."""
    try:
        assert _ZF is not None and _OUT_ROOT is not None and _MAX_LEN is not None and _OVERWRITE is not None
        info = _ZF.getinfo(name)
        p = PurePosixPath(name)
        rel_path = Path(*p.parts)
        crc_hex = f"{info.CRC:08x}"[-6:]
        rel_path = shorten_filename_only_if_needed(_OUT_ROOT, rel_path, _MAX_LEN, crc_hex)
        out_path = _OUT_ROOT / rel_path

        if out_path.exists() and not _OVERWRITE:
            try:
                if out_path.stat().st_size == info.file_size:
                    return (0, 1, 0)
            except Exception:
                pass

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with _ZF.open(info, 'r') as src, open(out_path, 'wb') as dst:
            for chunk in iter(lambda: src.read(4 * 1024 * 1024), b''):
                dst.write(chunk)

        try:
            if out_path.stat().st_size != info.file_size:
                out_path.unlink(missing_ok=True)
                return (0, 0, 1)
        except Exception:
            return (0, 0, 1)
        return (1, 0, 0)
    except Exception:
        return (0, 0, 1)

try:
    from tqdm import tqdm
except Exception:  # tqdm optional fall-back
    def tqdm(iterable: Iterable, total: int | None = None, desc: str | None = None, unit: str | None = None, **kwargs):
        return iterable


# Scan-type filter (case-insensitive)
SCAN_PATTERN = r"(mp-?rage|mprage|ir-?spgr|fspgr|bravo|3d\s*t1|sag.*t1)"


def shorten_filename_only_if_needed(base_dir: Path, rel_path: Path, max_total_len: int, crc_hex: str) -> Path:
    """
    Preserve the original directory structure exactly and only shorten the final filename
    if the full path exceeds max_total_len characters. Append a short CRC suffix to keep uniqueness.
    """
    full = base_dir / rel_path
    if len(str(full)) <= max_total_len:
        return rel_path

    parent = rel_path.parent
    stem = rel_path.stem
    suffix = rel_path.suffix  # keep original extension

    # Reserve for underscore + crc + suffix
    reserve = 1 + len(crc_hex) + len(suffix)
    # Compute available characters for the stem given base_dir and parent
    # +1 for path separator
    available = max(8, max_total_len - len(str(base_dir / parent)) - 1 - reserve)
    new_stem = stem[:available]
    new_name = f"{new_stem}_{crc_hex}{suffix}"
    return parent / new_name


def extract_preserving_dirs_and_filter(
    zip_path: str,
    extract_root: str,
    scan_regex: str = SCAN_PATTERN,
    max_total_len: int = 240,
    overwrite: bool = False,
    workers: int | None = None,
) -> None:
    """
    Re-extract relevant DICOMs, preserving full directory structure from the zip.
    Only modify (truncate) the final filename when Windows path length would be exceeded.

    - Filter files whose full zip path matches scan_regex and endswith .dcm
    - Keep all intermediate directories exactly as-is
    - Shorten only the final filename when needed, appending a CRC suffix
    - Overwrite existing files when overwrite=True, else skip if same size
    """
    zip_path_p = Path(zip_path)
    out_root = Path(extract_root)

    if not zip_path_p.exists():
        print(f"❌ Zip not found: {zip_path_p}")
        return

    out_root.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(scan_regex, re.IGNORECASE)

    print(f"📦 Zip: {zip_path_p}")
    print(f"📁 Extract to: {out_root}")
    print(f"🔎 Filter: {scan_regex}")

    # Build list of relevant names once
    with zipfile.ZipFile(zip_path_p, 'r') as zf:
        names = zf.namelist()
    relevant = [n for n in names if n.lower().endswith('.dcm') and pattern.search(n)]
    print(f"📊 Relevant files: {len(relevant):,}")

    # Decide worker count
    if workers is None:
        # Keep a safe default to avoid overwhelming disk
        workers = max(1, min(8, (cpu_count() or 2) - 1))
    print(f"🧵 Workers: {workers}")

    written = 0
    skipped = 0
    errors = 0

    if workers <= 1:
        # Serial path (original logic, slightly faster buffer)
        with zipfile.ZipFile(zip_path_p, 'r') as zf2:
            for name in tqdm(relevant, desc="Extracting", unit="files"):
                try:
                    p = PurePosixPath(name)
                    rel_path = Path(*p.parts)
                    info = zf2.getinfo(name)
                    crc_hex = f"{info.CRC:08x}"[-6:]
                    rel_path = shorten_filename_only_if_needed(out_root, rel_path, max_total_len, crc_hex)
                    out_path = out_root / rel_path

                    if out_path.exists() and not overwrite:
                        try:
                            if out_path.stat().st_size == info.file_size:
                                skipped += 1
                                continue
                        except Exception:
                            pass

                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf2.open(info, 'r') as src, open(out_path, 'wb') as dst:
                        for chunk in iter(lambda: src.read(4 * 1024 * 1024), b''):
                            dst.write(chunk)

                    try:
                        if out_path.stat().st_size != info.file_size:
                            out_path.unlink(missing_ok=True)
                            errors += 1
                        else:
                            written += 1
                    except Exception:
                        errors += 1
                except Exception:
                    errors += 1
    else:
        # Parallel path: each worker keeps its own ZipFile handle (Windows-safe)
        with Pool(processes=workers, initializer=_init_pool,
                  initargs=(str(zip_path_p), str(out_root), max_total_len, overwrite)) as pool:
            for w, s, e in tqdm(pool.imap_unordered(_extract_one, relevant, chunksize=128),
                                total=len(relevant), desc="Extracting", unit="files"):
                written += w
                skipped += s
                errors += e

    print("\n✅ Extraction finished")
    print(f"   ✔ Written: {written:,}")
    print(f"   ⏭ Skipped (same size): {skipped:,}")
    print(f"   ⚠ Errors: {errors:,}")


if __name__ == "__main__":
    # Defaults (edit as needed or pass via CLI)
    zip_file = r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\MRI\AD_CN_all_available_data.zip"
    extract_dir = r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\MRI\extracted_images_new"
    overwrite_flag = False  # set True to force re-write

    # CLI usage: python mri_extract_full.py <zip> <out_dir> [--overwrite] [--workers N]
    if len(sys.argv) >= 3:
        zip_file = sys.argv[1]
        extract_dir = sys.argv[2]
        if len(sys.argv) >= 4 and sys.argv[3] == "--overwrite":
            overwrite_flag = True
        # Optional 5th arg for workers
        if len(sys.argv) >= 6 and sys.argv[4] == "--workers":
            try:
                workers_arg = int(sys.argv[5])
            except Exception:
                workers_arg = None
        else:
            workers_arg = None
    else:
        workers_arg = None

    extract_preserving_dirs_and_filter(
        zip_path=zip_file,
        extract_root=extract_dir,
        scan_regex=SCAN_PATTERN,
        max_total_len=240,
        overwrite=overwrite_flag,
        workers=workers_arg,
    )


