"""Tests for the take_snapshot file rotation logic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

# rotation.py is loaded straight from its file instead of through
# custom_components.advanced_snapshot, whose __init__.py imports Home
# Assistant, Pillow and ffmpeg-python: dependencies this test does not need.
_ROTATION_PATH = (
    Path(__file__).resolve().parent.parent
    / "custom_components"
    / "advanced_snapshot"
    / "rotation.py"
)
_spec = importlib.util.spec_from_file_location("advanced_snapshot_rotation", _ROTATION_PATH)
_rotation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rotation)
async_rotate_snapshots = _rotation.async_rotate_snapshots


class FakeHass:
    """Stand-in for HomeAssistant: only async_add_executor_job is used."""

    async def async_add_executor_job(self, func, *args):
        """Call the function directly; there is no real event loop to protect here."""
        return func(*args)


def _write(path: Path, content: str) -> None:
    """Create a file with distinguishing content, so a rename can be verified."""
    path.write_text(content)


async def test_first_snapshot_has_nothing_to_rotate(tmp_path: Path) -> None:
    """With no earlier snapshot, rotation is a no-op that returns the base path."""
    target = tmp_path / "cam.jpg"

    result = await async_rotate_snapshots(FakeHass(), str(target), "cam", 3)

    assert result == str(target)
    assert list(tmp_path.iterdir()) == []


async def test_shifts_the_current_snapshot_into_slot_one(tmp_path: Path) -> None:
    """The existing base file becomes generation 1 before the new one is saved."""
    _write(tmp_path / "cam.jpg", "newest")

    result = await async_rotate_snapshots(FakeHass(), str(tmp_path / "cam.jpg"), "cam", 3)

    assert result == str(tmp_path / "cam.jpg")
    assert not (tmp_path / "cam.jpg").exists()
    assert (tmp_path / "cam-1.jpg").read_text() == "newest"


async def test_shifts_a_full_chain_up_by_one(tmp_path: Path) -> None:
    """Every existing generation moves up one slot, oldest last."""
    _write(tmp_path / "cam.jpg", "gen0")
    _write(tmp_path / "cam-1.jpg", "gen1")

    result = await async_rotate_snapshots(FakeHass(), str(tmp_path / "cam.jpg"), "cam", 3)

    assert result == str(tmp_path / "cam.jpg")
    assert not (tmp_path / "cam.jpg").exists()
    assert (tmp_path / "cam-1.jpg").read_text() == "gen0"
    assert (tmp_path / "cam-2.jpg").read_text() == "gen1"


async def test_deletes_the_oldest_generation_once_the_cap_is_reached(
    tmp_path: Path,
) -> None:
    """The generation at the cap is dropped instead of shifted further."""
    _write(tmp_path / "cam.jpg", "gen0")
    _write(tmp_path / "cam-1.jpg", "gen1")
    _write(tmp_path / "cam-2.jpg", "gen2-to-be-dropped")

    result = await async_rotate_snapshots(FakeHass(), str(tmp_path / "cam.jpg"), "cam", 2)

    assert result == str(tmp_path / "cam.jpg")
    assert not (tmp_path / "cam.jpg").exists()
    assert (tmp_path / "cam-1.jpg").read_text() == "gen0"
    assert (tmp_path / "cam-2.jpg").read_text() == "gen1"
    # The oldest generation is gone rather than shifted into a cam-3.jpg.
    assert set(p.name for p in tmp_path.iterdir()) == {"cam-1.jpg", "cam-2.jpg"}


async def test_a_gap_in_the_chain_does_not_raise(tmp_path: Path) -> None:
    """A missing intermediate generation (e.g. removed by hand) is skipped."""
    _write(tmp_path / "cam.jpg", "gen0")
    _write(tmp_path / "cam-2.jpg", "gen2")
    # cam-1.jpg is missing.

    result = await async_rotate_snapshots(FakeHass(), str(tmp_path / "cam.jpg"), "cam", 3)

    assert result == str(tmp_path / "cam.jpg")
    assert (tmp_path / "cam-1.jpg").read_text() == "gen0"
    assert (tmp_path / "cam-3.jpg").read_text() == "gen2"
