"""Historical, no-lookahead strategy replay utilities."""
from dataclasses import dataclass, asdict
from typing import Iterable, List, Dict, Optional

@dataclass
class TradeOutcome:
    entry: float
    stop: float
    target1: float
    target2: float
    target3: float
    max_high: float
    max_low: float
    exit_price: float
    result: str
    r_multiple: float

@dataclass
class BacktestResult:
    observations: int
    wins: int
    losses: int
    scratches: int
    hit_rate: float
    avg_r: float
    max_r: float
    max_drawdown_r: float
    results: List[TradeOutcome]

def replay_long(entry: float, stop: float, targets: list[float], bars: Iterable[dict]) -> TradeOutcome:
    highs=[]; lows=[]; exit_price=entry; result="SCRATCH"
    for b in bars:
        h=float(b["High"]); l=float(b["Low"]); highs.append(h); lows.append(l)
        if l <= stop:
            exit_price=stop; result="LOSS"; break
        # Conservative intra-bar ordering: stop wins ties over target.
        if h >= targets[2]: exit_price=targets[2]; result="TP3"; break
        if h >= targets[1]: exit_price=targets[1]; result="TP2"; break
        if h >= targets[0]: exit_price=targets[0]; result="TP1"; break
    risk=max(1e-9,entry-stop); r=(exit_price-entry)/risk
    return TradeOutcome(entry,stop,*targets,max(highs or [entry]),min(lows or [entry]),exit_price,result,round(r,3))

def backtest_long(setups: Iterable[Dict], bars_lookup: Dict[str, Iterable[dict]]) -> BacktestResult:
    results=[]
    for s in setups:
        bars=bars_lookup.get(s["id"],[])
        results.append(replay_long(float(s["entry"]),float(s["stop"]),[float(s["tp1"]),float(s["tp2"]),float(s["tp3"])],bars))
    if not results:return BacktestResult(0,0,0,0,0.0,0.0,0.0,0.0,[])
    wins=sum(r.r_multiple>0 for r in results); losses=sum(r.r_multiple<0 for r in results); scratches=len(results)-wins-losses
    eq=peak=dd=0.0
    for r in results:
        eq+=r.r_multiple; peak=max(peak,eq); dd=min(dd,eq-peak)
    return BacktestResult(len(results),wins,losses,scratches,round(wins/len(results)*100,2),round(sum(r.r_multiple for r in results)/len(results),3),round(max(r.r_multiple for r in results),3),round(dd,3),results)

def evaluate_returns(returns: Iterable[float], threshold: float=0.0) -> BacktestResult:
    vals=list(returns); wins=sum(r>threshold for r in vals); losses=sum(r<threshold for r in vals); scratches=len(vals)-wins-losses
    eq=peak=dd=0.0
    for r in vals:
        eq+=r; peak=max(peak,eq); dd=min(dd,eq-peak)
    return BacktestResult(len(vals),wins,losses,scratches,round(wins/len(vals)*100,2) if vals else 0.0,round(sum(vals)/len(vals),3) if vals else 0.0,round(max(vals),3) if vals else 0.0,round(dd,3),[])
