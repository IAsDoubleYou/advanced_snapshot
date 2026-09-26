"""File rotation for take_snapshot: keep at most N numbered snapshots.

Kept free of Home Assistant, Pillow and ffmpeg imports on purpose, so it can
be unit tested without pulling in the whole custom component's dependencies.
"""

from __future__ import annotations

import logging
import os
from typing import Any

_LOGGER = logging.getLogger(__name__)


async def async_rotate_snapshots(
    hass: Any, file_path: str, base_file_name: str, max_snapshots: int
) -> str:
    """Age out old snapshots and return the path the new one should be saved to.

    Keeps at most ``max_snapshots`` numbered files (``base-1.ext`` ..
    ``base-N.ext``) alongside the newest one under ``base_file_name.ext``: the
    oldest numbered file is deleted, the rest shift up by one, and the current
    base file (now one generation older) takes the first numbered slot.
    """
    target_dir = os.path.dirname(file_path)
    _, ext = os.path.splitext(file_path)
    base_full_path = os.path.join(target_dir, f"{base_file_name}{ext}")

    oldest_file_to_delete = os.path.join(
        target_dir, f"{base_file_name}-{max_snapshots}{ext}"
    )
    if await hass.async_add_executor_job(os.path.exists, oldest_file_to_delete):
        try:
            await hass.async_add_executor_job(os.remove, oldest_file_to_delete)
            _LOGGER.debug("Deleted oldest snapshot: %s", oldest_file_to_delete)
        except OSError as err:
            _LOGGER.warning(
                "Could not delete oldest snapshot %s: %s", oldest_file_to_delete, err
            )

    for i in range(max_snapshots - 1, 0, -1):
        old_rotated_path = os.path.join(target_dir, f"{base_file_name}-{i}{ext}")
        new_rotated_path = os.path.join(target_dir, f"{base_file_name}-{i + 1}{ext}")
        if await hass.async_add_executor_job(os.path.exists, old_rotated_path):
            try:
                await hass.async_add_executor_job(
                    os.rename, old_rotated_path, new_rotated_path
                )
                _LOGGER.debug("Renamed %s to %s", old_rotated_path, new_rotated_path)
            except OSError as err:
                _LOGGER.warning(
                    "Could not rename %s to %s: %s",
                    old_rotated_path,
                    new_rotated_path,
                    err,
                )

    if await hass.async_add_executor_job(os.path.exists, base_full_path):
        first_rotated_path = os.path.join(target_dir, f"{base_file_name}-1{ext}")
        try:
            await hass.async_add_executor_job(
                os.rename, base_full_path, first_rotated_path
            )
            _LOGGER.debug("Renamed %s to %s", base_full_path, first_rotated_path)
        except OSError as err:
            _LOGGER.warning(
                "Could not rename %s to %s: %s", base_full_path, first_rotated_path, err
            )

    return base_full_path
