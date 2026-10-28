# p2-monitor

Online monitoring of the P2 VMM readout. Uses
[p2-decode](https://github.com/akallitss/p2-decode) for hits and
[p2-mapping](https://github.com/akallitss/p2-mapping) for where each channel
sits, and adds only monitoring: counts, hit rates in the electronics and
physical views, data volume, the per-capture trend record, and rate plots.

## Use

```bash
p2-monitor reduce capture.pcapng --store-dir stores/cap1 --setup lab_saclay_2026
p2-monitor rates  stores/cap1              # per detector / connector / hybrid / VMM
p2-monitor plot   stores/ --out plots/     # merges every store under stores/
p2-monitor trend  run_dir/ --csv trend.csv # one row per store, by capture start
```

```python
from p2_monitor import process_capture, load_counts, channel_rates, store_mapping

scal = process_capture("cap.pcapng", "stores/cap1", setup="lab_saclay_2026")
scal["rates"]["per_detector"]["P2_IN"]      # {"rate_hz", "rate_hz_per_cm2", "live_channels", ...}
scal["volume"]                               # live time, hits/s, MB/min, per FEC
df = channel_rates(load_counts("stores/cap1"), scal["volume"]["live_time_s"],
                   store_mapping("stores/cap1"))   # one row per channel, dead ones included
```

The setup is resolved as in p2-mapping (`--setup`, `P2_SETUP`, machine
default); the cabling is picked at the capture's first packet unless `--when`
is given. `--no-mapping` gives the electronics view only.

## Store

Built under `<store>.partial` and renamed into place when complete, so a store
that exists is complete. A superset of the `vmm_reduce.py` store, so
`render_from_store.py` and the trend dashboard read it unchanged.

| File | Content |
| --- | --- |
| `<column>.npy`, `meta.json` | p2-decode hit columns and decoder meta (dropped with `--drop-columns`) |
| `counts.npz` | per-VMM histograms, additive across captures |
| `scalars.json` | trend record (`vmm_reduce` keys) + `rates` + `volume` + `capture_start`, `setup`, `versions` |
| `setup_resolved.json` | the mapping used; `store_mapping(store)` rebuilds it |

## What is measured

- **Rates** are hits / live time, the span of the capture's packet timestamps.
  Levels: `per_vmm`, `per_hybrid`, `per_connector` (`P2_IN:c4`), `per_half`
  (`P2_IN:c4:bot`), `per_detector`, each with hits, rate, channels, live
  channels, the hottest channel's rate, and Hz/cm² for the physical levels.
  Cabled channels that never fire are kept at rate 0. Hits on VMMs not cabled
  to a detector are counted under `uncabled`.
- **Data volume**: MB/min of the capture file (what `rate_guard` limits) and of
  the FEC payloads, hits/s and packets/s, per FEC.
- **Frame loss** is reported as not measurable: the ALINX firmware's SRS frame
  counter is constant, so lost frames cannot be counted from the data.

## Tests

```bash
pytest
P2_MONITOR_LEGACY=../P2_basket_online_analysis P2_MONITOR_CAPTURE=cap.pcapng \
    pytest tests/test_legacy.py
```

`test_legacy.py` reduces a real capture with both `vmm_reduce.process_file` and
p2-monitor (with `legacy_word_count=True`) and requires every `counts.npz`
array and every scalar the old reduce writes to be identical.

## Not yet here

- The efficiency stage (`vmm_efficiency.analyse`): it depends on the trigger
  channel and masks, which belong in the setup file first.
- The full per-capture QA plot set of `render_from_store.py`.
- A watcher: the DAQ's `vmm_processor_watcher.py` keeps running the reduce;
  switching it to `p2_monitor.process_capture` is the next step.
