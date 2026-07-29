"""Shared test setup.

Redirect the HuggingFace caches to a writable temp dir BEFORE any SUT import: ``datasets`` reads
``HF_DATASETS_CACHE`` / ``HF_HOME`` at import time, and ``load_dataset('parquet', ...)`` file-locks
inside that cache. The default ``~/.cache/huggingface`` may be read-only or shared, so point it at
a per-process temp dir (and keep the real cache clean). pytest imports ``conftest.py`` before the
test modules, which import ``datasets`` transitively via ``src.train.gpt``; ``mp.spawn`` children
inherit this env from the parent.
"""

from __future__ import annotations

import os
import tempfile

_tmp = tempfile.gettempdir()
os.environ.setdefault('HF_DATASETS_CACHE', os.path.join(_tmp, 'test_gpt_hf_datasets'))
os.environ.setdefault('HF_HOME', os.path.join(_tmp, 'test_gpt_hf_home'))
