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
p2-monitor watch  runs/ --setup lab_saclay_2026          # reduce each closed capture
p2-monitor live   runs/run_5/sub_00/raw_daq_data --setup lab_saclay_2026  # as data arrive
```

## Watching and live monitoring

Two ways to follow a run, which can run side by side:

| | `watch` | `live` |
| --- | --- | --- |
| Reads | each capture once dumpcap has closed it | the capture dumpcap is writing, every few seconds |
| Latency | about one rotation (44.4 s at SPS) | the poll interval (default 5 s) |
| Writes | a store per capture, `hits_store/latest.json` | `live.json` (rates over the last `--window` s), optional PNGs |
| For | the permanent record, trends, re-plotting | watching rates and volume while taking data |

`watch` uses the DAQ watcher's rule for a finished capture: size unchanged
since the previous poll, and a newer capture exists, the sub-run wrote
`.capture_done`, or no write for twice `--capture-duration`. Newest first; a
capture whose store has `scalars.json` is skipped, so it can be restarted at
any time. Each reduce is a child process at nice 19 with `--timeout`; a capture
that fails `--max-attempts` times is left alone. `latest.json` always describes
the newest capture, also while older ones are backfilled.

`live` follows the newest capture; when dumpcap rotates, it reads the old file
to its end and moves on. Only bytes appended since the previous poll are read
(`p2_decode.StreamDecoder`), with the decoder state carried over, so the hits
are exactly those of decoding the finished files. Rates are over the last
`--window` seconds of packet time; `data_age_s` in `live.json` shows how long
ago the newest packet arrived, so a stalled DAQ is visible at a glance.

`p2-monitor-reduce` is `reduce` with `vmm_reduce.py`'s calling convention, so the
DAQ's `vmm_processor_watcher.py` can use it as its worker (config key `worker`)
and keep its own memory-kill and stall protection.

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
| `counts.npz` | per-VMM histograms, additive across captures (the old layout) |
| `counts_chip.npz` | per-chip histograms, chip = `fec*32 + vmm`, uint64 |
| `scalars.json` | trend record (`vmm_reduce` keys) + `rates` + `volume` + `capture_start`, `setup`, `versions` |
| `setup_resolved.json` | the mapping used; `store_mapping(store)` rebuilds it |

## Scale and memory

Built for the full experiment (180 hybrids = 360 VMMs over 12+ FECs), not for
the lab's cosmic rates, and no limit assumes a data rate:

- **Chips are (fec, vmm).** A VMM id is 5 bits and repeats across FECs, so rates
  and histograms are per chip (`per_chip` keys `"fec:vmm"`, `counts_chip.npz`).
  `counts.npz` keeps the per-VMM layout for the existing QA readers; with
  several FECs it merges chips that share a VMM id.
- **reduce memory does not grow with the capture.** Histograms fill while
  decoding, in chunks of `--max-words` (default 500k): peak anonymous memory
  ~250 MB for a 141 MB capture and the same for a 425 MB one, measured. The
  capture itself is memory-mapped; those pages are reclaimable page cache.
  Each reduce under `watch` runs at nice 19 with `oom_score_adj` 500.
- **live is a sampling monitor.** It reads at most `--max-read-mb` (16) per poll,
  skips to the newest data when more than `--max-behind-mb` (64) is unread and
  reports `sampling.fraction_read`, keeps at most ~60 time buckets of channel
  occupancy, sheds its window above `--rss-limit-mb`, and sets its own
  `--oom-score-adj` (500). Measured: 286 MB peak catching up a whole 141 MB
  file, 192 MB with skip-ahead.
- Keeping the hit columns costs ~3.4x the capture on disk; `--drop-columns`
  keeps only the histograms and scalars.

## What is measured

- **Rates** are hits / live time, the span of the capture's packet timestamps.
  Levels: `per_fec`, `per_chip` (`"12:2"`), `per_hybrid`, `per_connector`
  (`P2_IN:c4`), `per_half` (`P2_IN:c4:bot`), `per_detector`, each with hits,
  rate, channels, live channels, the hottest channel's rate, and Hz/cm² for
  the physical levels. Cabled channels that never fire are kept at rate 0.
  Hits on chips not cabled to a detector are counted under `uncabled`.
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
- The DAQ's own watcher still runs `vmm_reduce.py`; pointing its `worker` at
  `p2-monitor-reduce`, or replacing it with `p2-monitor watch`, is the switch-over.

## Versions

- **0.7.1** analyze and qa merge stores one at a time: peak memory ~flat with the
  number of captures (1 store 298 MB, 30 stores 353 MB).
- **0.7.0** `p2-monitor analyze <run> --out-dir DIR`: the detector view of a run,
  per sub-run and for the whole run, from the stored histograms only (common to
  every campaign): rate on the detector surface, per-channel rates, rates per
  level, per-VMM ADC per channel (map + mean), amplitude per detector and
  connector half, rate vs time per detector, sub-run trend of rate and mean ADC,
  and summary.json (hits, rate, Hz/cm2, live/dead/noisy channels, mean ADC).
  Stores are redrawn with the setup FILE they were reduced with (setup_file in
  setup_resolved.json), not whatever the bare name resolves to now.
- **0.6.1** QA ADC axis range from the widest single chip, not the sum (a hot
  chip's pedestal cut the other chips' tails).
- **0.6.0** `p2-monitor qa <store>... --out-dir DIR [--events-json]` (and the
  drop-in `p2-monitor-qa`): the per-capture QA plots drawn from histograms, with
  vmm_pcapng_qa.py's file names and events.json: ADC, ADC by over-threshold,
  ADC vs channel, OT fraction, channel occupancy, hits per chip, BCID, TDC,
  frame offset, hit time in frame, and hits per second through the capture
  (by packet time). Per chip (FEC:VMM); several stores are merged. The reduce
  now fills the needed histograms (`chip_adc_ot1`, `chip_adc_vs_ch`,
  `chip_bcid`, `chip_tdc`, `chip_offset`, `chip_ts`, `rate_s`) in
  counts_chip.npz, so hit columns are no longer needed for plots. Older stores
  must be re-reduced. Needs p2-decode 0.8.1.
- **0.5.0** `live --udp HOST:PORT` reads the FEC's stream from the network,
  no file (p2-decode 0.8.0). dumpcap is unaffected: the kernel gives each
  datagram to both. Each poll reads at most `--max-read-mb`; the kernel drops
  and counts the rest. In both modes rates and volume are now divided by live
  time x `sampling.window_fraction` (packets decoded / arrived, counting
  kernel drops or file skips), i.e. scaled to the full stream: before, a file
  monitor that skipped ahead under a flood under-reported its rates.
  `live.json` gains `source`. Console lines are flushed (readable via a pipe).
- **0.4.2** same code as the v0.4.1 tag, whose `__version__` still read 0.4.0.
- **0.4.1** `reduce --mapping-optional`: a capture no cabling entry covers is
  reduced in electronics coordinates, with the reason in `mapping_error`; `live`
  always does so and tries the mapping once per sub-run.
- **0.4.0** `live --follow-newest <runs_dir>`: one live process for a whole session,
  moving to the newest sub-run by itself (`run`/`subrun` in live.json).
- **0.3.1** with `--drop-columns` only `vmm` and `time` are written during the
  reduce (needs p2-decode 0.7.0); measured on the lab box below.
- **0.3.0** per-chip `(fec, vmm)` rates and `counts_chip.npz`; single-pass reduce
  with bounded memory (`--max-words`); exact detector ADC percentiles from
  histograms; `live` limits (`--max-read-mb`, `--max-behind-mb` skip-ahead,
  `--rss-limit-mb`, `--oom-score-adj`) and bounded window; `watch` children
  get `oom_score_adj` 500; plots scale to many chips and detectors. Rate keys
  changed: `per_vmm` is now `per_chip`. Needs p2-decode 0.6.1, p2-mapping 0.3.0.
- **0.2.0** `watch` (closed captures, DAQ finalisation rules, `latest.json`),
  `live` (streaming rates over a sliding window, `live.json`, PNGs),
  `p2-monitor-reduce`, `reduce --calibration`. Needs p2-decode 0.4.0.
- **0.1.1** rate plots when no detector fired.
- **0.1.0** reduce, rates, volume, plots, trend.
