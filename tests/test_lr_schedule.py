"""LR schedule boundary-value tests. Monkeypatches train.py's module-level
WARMUP_RATIO/WARMDOWN_RATIO/FINAL_LR_FRAC constants to known values rather
than asserting against whatever the autoresearch loop currently has them
set to (those are exactly the constants the loop mutates every experiment),
so these tests stay meaningful regardless of the loop's current tuning."""
import train


def test_zero_warmup_jumps_straight_to_full_lr_at_progress_zero(monkeypatch):
    """This is the exact bug we found in nanochat's own chat_sft.py:
    warmup_ratio=0.0 means progress=0 already hits the flat-LR branch,
    giving lrm=1.0 with no ramp at all. Documenting it here as a known
    dangerous configuration, not proposing to forbid it in train.py."""
    monkeypatch.setattr(train, "WARMUP_RATIO", 0.0)
    monkeypatch.setattr(train, "WARMDOWN_RATIO", 0.5)
    monkeypatch.setattr(train, "FINAL_LR_FRAC", 0.0)
    assert train.get_lr_multiplier(0.0) == 1.0


def test_nonzero_warmup_ramps_from_near_zero(monkeypatch):
    monkeypatch.setattr(train, "WARMUP_RATIO", 0.1)
    monkeypatch.setattr(train, "WARMDOWN_RATIO", 0.5)
    monkeypatch.setattr(train, "FINAL_LR_FRAC", 0.0)
    lrm_at_zero = train.get_lr_multiplier(0.0)
    lrm_at_half_warmup = train.get_lr_multiplier(0.05)
    assert lrm_at_zero < 0.01
    assert 0.4 < lrm_at_half_warmup < 0.6


def test_flat_region_holds_at_peak(monkeypatch):
    monkeypatch.setattr(train, "WARMUP_RATIO", 0.1)
    monkeypatch.setattr(train, "WARMDOWN_RATIO", 0.5)
    monkeypatch.setattr(train, "FINAL_LR_FRAC", 0.0)
    assert train.get_lr_multiplier(0.1) == 1.0
    assert train.get_lr_multiplier(0.3) == 1.0
    assert train.get_lr_multiplier(0.5) == 1.0


def test_warmdown_reaches_final_lr_frac_at_progress_one(monkeypatch):
    monkeypatch.setattr(train, "WARMUP_RATIO", 0.05)
    monkeypatch.setattr(train, "WARMDOWN_RATIO", 0.8)
    monkeypatch.setattr(train, "FINAL_LR_FRAC", 0.045)
    assert abs(train.get_lr_multiplier(1.0) - 0.045) < 1e-9


def test_warmdown_is_monotonically_decreasing(monkeypatch):
    monkeypatch.setattr(train, "WARMUP_RATIO", 0.05)
    monkeypatch.setattr(train, "WARMDOWN_RATIO", 0.8)
    monkeypatch.setattr(train, "FINAL_LR_FRAC", 0.045)
    samples = [train.get_lr_multiplier(p) for p in [0.2, 0.4, 0.6, 0.8, 1.0]]
    assert samples == sorted(samples, reverse=True)
