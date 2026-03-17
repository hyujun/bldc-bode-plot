"""
output_manager.py — Timestamped output directory management.

Creates YYMMDD_HHMM/ directories with subdirectories for plots, data,
and exports.  Enforces a maximum of 10 result directories.
"""

import os
import re
import shutil
from datetime import datetime

from .config import logger

_DIR_PATTERN = re.compile(r"^\d{6}_\d{4}$")
_MAX_DIRS = 10

_SUBDIRS = {
    ".png": "plots",
    ".npz": "data",
    ".csv": "export",
    ".json": "export",
}


class OutputManager:
    """Create and manage a timestamped output directory.

    Structure
    ---------
      YYMMDD_HHMM/
      ├── plots/    — .png visualization files
      ├── data/     — .npz raw data files
      └── export/   — .csv, .json exported files
    """

    def __init__(self, base_dir: str = ".", create: bool = True):
        self.base_dir = os.path.abspath(base_dir)
        if create:
            stamp = datetime.now().strftime("%y%m%d_%H%M")
            self.root = os.path.join(self.base_dir, stamp)
            os.makedirs(self.root, exist_ok=True)
            for sub in set(_SUBDIRS.values()):
                os.makedirs(os.path.join(self.root, sub), exist_ok=True)
            self._enforce_max_dirs()
        else:
            self.root = self.base_dir

    @classmethod
    def from_existing(cls, folder_path: str) -> "OutputManager":
        """Wrap an existing output directory (for --reanalyze)."""
        obj = cls.__new__(cls)
        obj.base_dir = os.path.dirname(os.path.abspath(folder_path))
        obj.root = os.path.abspath(folder_path)
        for sub in set(_SUBDIRS.values()):
            os.makedirs(os.path.join(obj.root, sub), exist_ok=True)
        return obj

    def path(self, filename: str) -> str:
        """Return full path for a file, routed to the correct subdirectory."""
        ext = os.path.splitext(filename)[1].lower()
        sub = _SUBDIRS.get(ext, "")
        return os.path.join(self.root, sub, filename)

    def log_structure(self) -> None:
        """Log the output directory tree."""
        logger.info(f"Output directory: {self.root}")
        for subdir in sorted(set(_SUBDIRS.values())):
            d = os.path.join(self.root, subdir)
            if os.path.isdir(d):
                files = os.listdir(d)
                if files:
                    logger.info(f"  {subdir}/ ({len(files)} files)")

    def _enforce_max_dirs(self) -> None:
        """Delete oldest YYMMDD_HHMM directories beyond _MAX_DIRS."""
        dirs = []
        for name in os.listdir(self.base_dir):
            full = os.path.join(self.base_dir, name)
            if os.path.isdir(full) and _DIR_PATTERN.match(name):
                dirs.append(full)

        if len(dirs) <= _MAX_DIRS:
            return

        dirs.sort(key=lambda d: os.path.getmtime(d))
        to_remove = dirs[: len(dirs) - _MAX_DIRS]
        for d in to_remove:
            logger.info(f"Removing old output directory: {os.path.basename(d)}")
            shutil.rmtree(d, ignore_errors=True)
