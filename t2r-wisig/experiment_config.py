#!/usr/bin/env python3
"""
experiment_config.py  —  Central configuration dataclass.

Current protocol
----------------
Phase 2 re-initialises from the Phase 1 model at every fine-tune step by
default, and uses 1600 labelled samples per class per fine-tune day.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Optional, Union


NEU_DAY_DIRS = {
    1: 'day1/equ',
    2: 'day2/equ',
    3: 'day3/equ',
    4: 'day4/equ',
    5: 'day5/equ',
    6: 'day6/equ',
    7: 'day7/equ',
    8: 'day8/equ',
    9: 'day9/equ',
}

PYCOM_DAY_DIRS = {
    1: 'Day_1',
    2: 'Day_2',
    3: 'Day_3',
    4: 'Day_4',
    5: 'Day_5',
}


@dataclass
class ExperimentConfig:
    # ── Paths ─────────────────────────────────────────────────────────────────
    data_root:    str   = os.environ.get('T2R_DATA_ROOT', './data/neu')
    output_root:  str   = 'res_out'
    file_key:     str   = '*.bin'
    location:     str   = ''
    dataset_layout: str  = 'auto'

    # ── Device split ──────────────────────────────────────────────────────────
    n_known:     int = 16
    n_val_dev:   int = 0
    n_unknown:   int = 4
    # Rotates which devices are unknown: with shift s, the device list is
    # rotated by s before splitting, so shift=0 → unknown {16..19},
    # shift=4 → unknown {0..3}, shift=8 → unknown {4..7}. Used to verify
    # results are not specific to one known/unknown split. Requires a full
    # pipeline run per shift (known devices define Phase-1 training).
    device_shift: int = 0
    wisig_rx_index: int  = 0
    wisig_full_receiver: str = 'node24-16'
    wisig_equalized: int = 0


    @property
    def _device_order(self) -> List[int]:
        total = self.n_known + self.n_val_dev + self.n_unknown
        return [(i + self.device_shift) % total for i in range(total)]

    @property
    def known_ids(self) -> List[int]:
        return sorted(self._device_order[:self.n_known])

    @property
    def val_device_ids(self) -> List[int]:
        return sorted(
            self._device_order[self.n_known:self.n_known + self.n_val_dev]
        )

    @property
    def unknown_ids(self) -> List[int]:
        return sorted(self._device_order[self.n_known + self.n_val_dev:])

    @property
    def all_known_ids(self) -> List[int]:
        return self.known_ids + self.val_device_ids

    # ── Session split defaults ────────────────────────────────────────────────
    # Dataset-specific scripts override these values for the ICC experiments.
    init_day:        int        = 1
    finetune_days:   List[int]  = field(default_factory=lambda: [2, 3, 4, 5, 6, 7])
    traj_day:        int        = 7
    test_day:        int        = 8

    # ── Data loading ──────────────────────────────────────────────────────────
    num_slice:       int   = 3000
    slice_len:       int   = 288
    window:          int   = 64
    data_type:       str   = 'IQ'
    mul_trans:       bool  = True
    start_idx:       int   = 0
    stride:          Union[int, str] = 288   # int, 'r' (random), or 'spread'

    # ── Model ─────────────────────────────────────────────────────────────────
    model_type:      str   = 'DF'
    emb_size:        int   = 128
    activation:      str   = 'elu'
    dropout_conv:    float = 0.3
    dropout_fc:      float = 0.4
    l2_reg:          float = 1e-4

    # ── Signal preprocessing ──────────────────────────────────────────────────
    use_preprocessing: bool  = True
    use_augmentation:  bool  = True
    aug_phase_rot:     float = 3.14159
    aug_amp_db:        float = 2.0   # reduced from 3.0: less destructive augmentation
    aug_snr_db:        float = 20.0  # slightly harder noise
    aug_prob:          float = 0.7
    aug_multipath_taps: int   = 0     # 0 disables; 3-5 simulates indoor echoes
    aug_multipath_mag:  float = 0.0   # max echo magnitude vs direct path

    # 'random' keeps the historical within-day shuffle (train/test slices can
    # share a recording, which inflates in-day accuracy). 'capture' splits
    # train and test across different capture files.
    ft_split_mode:     str   = 'random'

    # ── Sequential fine-tuning ────────────────────────────────────────────────
    ft_n_train:       int   = 1600
    ft_n_repeats:     int   = 1

    ft_always_reinit: bool  = False

    # ── Temporal trajectory ───────────────────────────────────────────────────
    traj_ewma:      float          = 0.8
    traj_threshold: Optional[float] = None

    # ── Misc ──────────────────────────────────────────────────────────────────
    verbose:        int   = 1
    seed:           int   = 42

    def __post_init__(self):
        self.finetune_days = sorted(self.finetune_days)
        self.dataset_layout = self._resolve_dataset_layout()
        if self.dataset_layout == 'pycom_indoor':
            if self.file_key == '*.bin':
                self.file_key = '*.dat'
            if self.location == 'equ':
                self.location = ''
            # These captures are one continuous 50M-sample recording per
            # transmission. A contiguous numeric stride reads <0.4% of it,
            # so every slice shares one channel/AGC realisation; spreading
            # the offsets across the whole capture is what makes the
            # train/test split (and cross-day evaluation) meaningful.
            if self.stride == 288:
                self.stride = 'spread'

        if self.dataset_layout == 'neu' and os.environ.get('T2R_SPREAD', '') == '1':
            # Contiguous sampling reads only the first ~0.7 MB of each ~100 MB
            # capture, so every slice shares one channel/AGC realisation and a
            # model can score well by recognising capture state. Spreading the
            # offsets across the whole recording is what makes a
            # capture-disjoint split meaningful (same reasoning as Pycom).
            self.stride = 'spread'

        configured_days = [self.init_day, self.traj_day, self.test_day]
        configured_days.extend(self.finetune_days)
        supported_days = self._day_dirs()
        unsupported = sorted({d for d in configured_days if d not in supported_days})
        if unsupported:
            raise ValueError(
                f"Unsupported day(s) {unsupported} for dataset_layout={self.dataset_layout!r}. "
                f"Supported days are {sorted(supported_days)}."
            )
        if self.test_day in self.finetune_days:
            raise ValueError(
                f"test_day={self.test_day} is in finetune_days={self.finetune_days}. "
                "Remove test_day from finetune_days to prevent data leakage."
            )
        if (self.finetune_days and max(self.finetune_days) >= self.test_day
                and os.environ.get('T2R_ALLOW_ANY_ORDER', '') != '1'):
            # Day indices are directory names, not capture order: on the NEU
            # corpus day7 and day8 hold the same recordings. Leakage is
            # therefore controlled by explicit twin exclusion, verified by
            # hashing, rather than by index ordering. Set
            # T2R_ALLOW_ANY_ORDER=1 for designs that need it.
            raise ValueError(
                f"max(finetune_days)={max(self.finetune_days)} >= test_day={self.test_day}. "
                "All finetune days must be strictly before test_day."
            )

    # ── Derived paths ─────────────────────────────────────────────────────────
    def _resolve_dataset_layout(self) -> str:
        if self.dataset_layout != 'auto':
            return self.dataset_layout
        if os.path.isdir(os.path.join(self.data_root, 'Day_1')):
            return 'pycom_indoor'
        return 'neu'

    def _day_dirs(self) -> dict[int, str]:
        if self.dataset_layout == 'neu':
            return NEU_DAY_DIRS
        if self.dataset_layout == 'pycom_indoor':
            return PYCOM_DAY_DIRS
        if self.dataset_layout in ('wisig_manysig', 'wisig_full'):
            # WiSig ships pickles rather than directories; days index capture
            # dates and are resolved by the loader, not by a path.
            return {1: '', 2: '', 3: '', 4: ''}
        raise ValueError(
            f"Unknown dataset_layout={self.dataset_layout!r}. Use 'auto', "
            "'neu', 'pycom_indoor', 'wisig_manysig' or 'wisig_full'."
        )

    def day_path(self, day: int) -> str:
        day_dirs = self._day_dirs()
        if day not in day_dirs:
            raise ValueError(
                f"Unsupported day={day} for dataset_layout={self.dataset_layout!r}. "
                f"Supported days are {sorted(day_dirs)}."
            )
        return os.path.join(self.data_root, day_dirs[day])

    @property
    def model_dir(self) -> str:
        p = os.path.join(self.output_root, 'modelDir')
        os.makedirs(p, exist_ok=True)
        return p

    @property
    def results_dir(self) -> str:
        os.makedirs(self.output_root, exist_ok=True)
        return self.output_root

    def summary(self) -> str:
        lines = [
            '─' * 60,
            '  ExperimentConfig',
            '─' * 60,
            f'  data_root         : {self.data_root}',
            f'  dataset_layout    : {self.dataset_layout}',
            f'  file_key/location : {self.file_key} / {self.location or "<none>"}',
            f'  known_ids         : {self.known_ids}  ({self.n_known} devices)',
            f'  val_device_ids    : {self.val_device_ids}  ({self.n_val_dev} devices)',
            f'  unknown_ids       : {self.unknown_ids}  ({self.n_unknown} devices)',
            f'  all_known_ids     : {self.all_known_ids}  ({len(self.all_known_ids)} devices)',
            f'  init_day          : {self.init_day}',
            f'  finetune_days     : {self.finetune_days}',
            f'  traj_day          : {self.traj_day}',
            f'  test_day          : {self.test_day}',
            f'  num_slice         : {self.num_slice}',
            f'  ft_n_train        : {self.ft_n_train}/class',
            f'  ft_always_reinit  : {self.ft_always_reinit}',
            f'  traj_ewma         : {self.traj_ewma}',
            f'  traj_threshold    : {self.traj_threshold or "auto-sweep"}',
            '─' * 60,
        ]
        return '\n'.join(lines)
