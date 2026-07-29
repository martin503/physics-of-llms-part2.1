"""Tests for the packing functions (``src.data.pack`` + ``pack_sequences``).

Packing feeds training directly: a non-uniform window or an off-by-one silently breaks
``PackedDataset`` and the collator (which asserts ``(batch, context_length)``). These are
pure list/parquet helpers with no iGSM dependency, so they run fast and always.

* ``pack_sequences``  -- the paper's "concatenate + right-truncate" packing (training path).
* ``pack_single``     -- one problem per window, EOS-padded (eval/dummy-model path).
* ``_write_shards``   -- uniform-length invariant + ``batch_NNNNNN.parquet`` naming.
* ``_load_streams``   -- eval-parquet column reader used by the ``pack`` CLI.
* the ``pack`` CLI    -- ``--mode packed``/``--mode single`` dispatch.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from src.data.igsm import EOS, pack_sequences
from src.data.pack import _load_streams, _write_shards, app, pack_single

CTX = 16


def _streams() -> list[list[int]]:
    """A few variable-length token lists (not valid iGSM layout -- packing doesn't care)."""
    return [[1, 2, 3], [4, 5], list(range(6, 14)), [99], [100, 101, 102, 103, 104]]


# --------------------------------------------------------------------------- #
# pack_sequences: concatenate + right-truncate
# --------------------------------------------------------------------------- #


def test_pack_sequences_uniform_windows():
    """Every output window is exactly ``context_length`` (the collator asserts this shape)."""
    windows = pack_sequences(_streams(), context_length=CTX)
    assert windows
    assert all(len(w) == CTX for w in windows)


def test_pack_sequences_drops_tail_and_preserves_prefix():
    """The partial tail is dropped; kept tokens equal ``flat[:n]`` exactly (in order)."""
    streams = _streams()
    flat = [t for s in streams for t in s]
    n = (len(flat) // CTX) * CTX
    windows = pack_sequences(streams, context_length=CTX)
    assert [t for w in windows for t in w] == flat[:n]
    assert len(flat) - n == len(flat) % CTX  # exactly the remainder is dropped


def test_pack_sequences_exact_multiple():
    """When total length is an exact multiple of ctx, no tokens are dropped."""
    streams = [list(range(CTX)), list(range(CTX, 2 * CTX))]
    windows = pack_sequences(streams, context_length=CTX)
    assert [t for w in windows for t in w] == list(range(2 * CTX))
    assert len(windows) == 2


def test_pack_sequences_empty_and_zero_ctx():
    """Empty input -> no windows; non-positive ctx is rejected by the guard assert."""
    assert pack_sequences([], context_length=CTX) == []
    assert pack_sequences([[1, 2, 3]], context_length=CTX) == []  # shorter than ctx -> dropped
    with pytest.raises(AssertionError):
        pack_sequences([[1]], context_length=0)


# --------------------------------------------------------------------------- #
# pack_single: one problem per window, EOS right-padded
# --------------------------------------------------------------------------- #


def test_pack_single_eos_padded_one_per_window():
    """Each window is exactly ctx, starts with EOS (BOS stand-in), and pads the trailing EOS."""
    streams = [[10, 11, 12], [20], list(range(30, 38))]  # all len+1 <= CTX(16)
    windows = pack_single(streams, CTX)
    assert len(windows) == len(streams)
    for s, w in zip(streams, windows, strict=True):
        assert len(w) == CTX
        assert w[0] == EOS
        assert w[1 : 1 + len(s)] == s  # the problem body sits right after the leading EOS
        trailing = CTX - (len(s) + 1)
        assert w[-trailing:] == [EOS] * trailing  # rest is EOS padding


def test_pack_single_drops_oversized():
    """Problems that cannot fit one window (``len + 1 > ctx``) are dropped, not errored."""
    fitting = [[1, 2, 3]]
    oversized = [list(range(CTX))]  # len CTX -> +1 leading EOS = CTX+1 > ctx -> dropped
    windows = pack_single(fitting + oversized, CTX)
    assert len(windows) == 1  # only the fitting problem survives


def test_pack_single_empty_when_all_oversized():
    """All problems oversized (or no input) -> no windows, no exception."""
    assert pack_single([list(range(CTX + 5))], CTX) == []
    assert pack_single([], CTX) == []


# --------------------------------------------------------------------------- #
# shard writer / loader (parquet I/O)
# --------------------------------------------------------------------------- #


def test_write_shards_uniform_and_naming(tmp_path):
    """``_write_shards`` enforces uniform window lengths, names shards ``batch_NNNNNN``, and
    returns ``ceil(len / shard_size)`` shards."""
    windows = [[1] * CTX, [2] * CTX, [3] * CTX]
    n = _write_shards(windows, tmp_path, shard_size=2)  # -> 2 shards (2 + 1)
    assert n == 2
    names = sorted(p.name for p in tmp_path.glob('batch_*.parquet'))
    assert names == ['batch_000000.parquet', 'batch_000001.parquet']
    # uniform-length invariant is enforced
    with pytest.raises(AssertionError):
        _write_shards([[1] * CTX, [2] * (CTX - 1)], tmp_path / 'sub', shard_size=10)
    # empty input is rejected
    with pytest.raises(AssertionError):
        _write_shards([], tmp_path / 'empty', shard_size=10)


def test_load_streams_round_trip(tmp_path):
    """``_load_streams`` reads the named column from every matching parquet back to list[list]."""
    col = [[1, 2, 3], [4, 5], [6]]
    pq.write_table(
        pa.table({'gold_token_id': col, 'op': [1, 2, 3]}), str(tmp_path / 'op_eq15.parquet')
    )
    pq.write_table(pa.table({'gold_token_id': [[7, 8]]}), str(tmp_path / 'op_eq20.parquet'))
    streams = _load_streams(tmp_path, 'gold_token_id', '*.parquet')
    assert streams == col + [[7, 8]]
    # a missing column / glob raises clearly
    with pytest.raises(FileNotFoundError):
        _load_streams(tmp_path, 'gold_token_id', 'nope_*.parquet')


# --------------------------------------------------------------------------- #
# pack CLI (--mode packed / --mode single dispatch)
# --------------------------------------------------------------------------- #


def _write_eval_dir(tmp_path, streams: list[list[int]]) -> str:
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({'gold_token_id': streams}), str(tmp_path / 'op_eq15.parquet'))
    return str(tmp_path)


def test_pack_cli_single_and_packed_modes(tmp_path):
    """Both modes produce uniform-length windows; ``single`` isolates one problem per window."""
    runner = CliRunner()
    in_dir = _write_eval_dir(tmp_path / 'in', [list(range(10)), list(range(5))])

    out_single = tmp_path / 'out_single'
    res = runner.invoke(
        app, ['--in', in_dir, '--out', str(out_single), '--ctx', '64', '--mode', 'single']
    )
    assert res.exit_code == 0, res.output
    wins = pq.read_table(out_single / 'batch_000000.parquet')['input_ids'].to_pylist()
    assert wins and all(len(w) == 64 for w in wins)
    assert all(w[0] == EOS for w in wins)  # single mode prepends EOS

    out_packed = tmp_path / 'out_packed'
    res = runner.invoke(
        app, ['--in', in_dir, '--out', str(out_packed), '--ctx', '8', '--mode', 'packed']
    )
    assert res.exit_code == 0, res.output
    wins = pq.read_table(out_packed / 'batch_000000.parquet')['input_ids'].to_pylist()
    assert wins and all(len(w) == 8 for w in wins)


def test_pack_cli_rejects_invalid_mode(tmp_path):
    """An unknown ``--mode`` is rejected with a non-zero exit (typer.BadParameter)."""
    runner = CliRunner()
    in_dir = _write_eval_dir(tmp_path / 'in', [[1, 2, 3]])
    res = runner.invoke(app, ['--in', in_dir, '--out', str(tmp_path / 'out'), '--mode', 'bogus'])
    assert res.exit_code != 0
