"""p2-monitor: online monitoring of the P2 VMM readout.

Uses p2-decode for hits and p2-mapping for where each channel sits, and adds
only monitoring: counts, rates in the electronics and physical views, data
volume, and the per-capture trend record.

    from p2_monitor import process_capture, channel_rates, trend_table
    scal = process_capture("cap.pcapng", "stores/cap", setup="lab_saclay_2026")
"""

__version__ = "0.7.0"

from .counts import (CountsAccumulator, build_chip_counts,           # noqa: E402
                     build_counts, chip_occupancy, merge_counts)
from .live import LiveMonitor                                        # noqa: E402
from .rates import channel_rates, summarize, volume                  # noqa: E402
from .reduce import capture_start, process_capture                   # noqa: E402
from .scalars import (add_detector_scalars,                          # noqa: E402
                      add_detector_scalars_from_counts, build_scalars,
                      percentile_from_hist)
from .store import (find_stores, load_counts, load_scalars,          # noqa: E402
                    merged_counts, store_mapping, trend_table)
from .watch import Watcher, finalized_captures, parse_capture_name   # noqa: E402

__all__ = [
    "process_capture", "capture_start", "build_counts", "merge_counts",
    "CountsAccumulator", "build_chip_counts", "chip_occupancy",
    "add_detector_scalars_from_counts", "percentile_from_hist",
    "build_scalars", "add_detector_scalars", "channel_rates", "summarize",
    "volume", "find_stores", "load_counts", "load_scalars", "merged_counts",
    "store_mapping", "Watcher", "LiveMonitor", "finalized_captures",
    "parse_capture_name",
    "trend_table", "__version__",
]
