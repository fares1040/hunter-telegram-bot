"""Stage 6 Adaptive Hunt - Track Record informed ranking (read-only)."""
import time
from typing import Dict

BASELINE_WIN_RATE = 0.50
MIN_SAMPLE = 20
DEVIATION_THRESHOLD = 0.15
MAX_ADJUST = 5
CACHE_TTL = 60.0

_cache: Dict[str, tuple] = {}
_cache_ts: float = 0

def _aggregate(memory) -> Dict[str, Dict[str, float]]:
    # Use list_signals to aggregate by setup_name for WON/LOST
    try:
        rows = memory.list_signals(limit=500)
    except Exception:
        return {}
    agg: Dict[str, Dict[str, float]] = {}
    for r in rows:
        if r.get("status") not in ("WON","LOST"):
            continue
        ret = r.get("forward_return") if r.get("forward_return") is not None else r.get("realized_return")
        if ret is None:
            continue
        setup = r.get("setup_name") or "UNKNOWN"
        if setup not in agg:
            agg[setup] = {"wins":0, "total":0}
        agg[setup]["total"] += 1
        if ret > 0:
            agg[setup]["wins"] += 1
    out={}
    for k,v in agg.items():
        if v["total"] >= MIN_SAMPLE:
            win_rate = v["wins"]/v["total"]
            dev = win_rate - BASELINE_WIN_RATE
            if abs(dev) > DEVIATION_THRESHOLD:
                adj = MAX_ADJUST if dev > 0 else -MAX_ADJUST
                # scale slightly by deviation but capped
                out[k] = {"win_rate": win_rate, "total": v["total"], "adjust": adj}
    return out

def get_adjustments(memory, force: bool = False) -> Dict[str, int]:
    global _cache, _cache_ts
    now = time.time()
    if not force and _cache and (now - _cache_ts) < CACHE_TTL:
        return _cache
    data = _aggregate(memory)
    res = {k: v["adjust"] for k,v in data.items()}
    _cache = res
    _cache_ts = now
    return res

def adjustment_for(setup_name: str, memory) -> int:
    if not setup_name or setup_name == "UNKNOWN":
        return 0
    adj = get_adjustments(memory).get(setup_name, 0)
    return adj

def clear_cache():
    global _cache, _cache_ts
    _cache = {}
    _cache_ts = 0
