"""p2-monitor: online monitoring of the P2 VMM readout.

Uses p2-decode for hits and p2-mapping for where each channel sits, and adds
only monitoring: counts, rates in the electronics and physical views, data
volume, and the per-capture trend record.

    from p2_monitor import process_capture, channel_rates, trend_table
    scal = process_capture("cap.pcapng", "stores/cap", setup="lab_saclay_2026")
"""

__version__ = "0.1.0"

from .counts import build_counts, merge_counts                       # noqa: E402
from .rates import channel_rates, summarize, volume                  # noqa: E402
from .reduce import capture_start, process_capture                   # noqa: E402
from .scalars import add_detector_scalars, build_scalars             # noqa: E402
from .store import (find_stores, load_counts, load_scalars,          # noqa: E402
                    merged_counts, store_mapping, trend_table)

__all__ = [
    "process_capture", "capture_start", "build_counts", "merge_counts",
    "build_scalars", "add_detector_scalars", "channel_rates", "summarize",
    "volume", "find_stores", "load_counts", "load_scalars", "merged_counts",
    "store_mapping",
    "trend_table", "__version__",
]
