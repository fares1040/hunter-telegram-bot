"""Options Intelligence Engine. Chain data is observational, not proof of flow."""
from datetime import date
from typing import Optional
from models.options import OptionsSnapshot, OptionsFlowProfile, OptionContract

class OptionsEngine:
    def analyze(self, snapshot: Optional[OptionsSnapshot], price: Optional[float], bullish: bool = True) -> OptionsFlowProfile:
        if not snapshot or not snapshot.contracts or not price:
            return OptionsFlowProfile(notes=["Options chain unavailable"], confidence=0, source="none")
        p = OptionsFlowProfile(source=snapshot.source, inferred=True)
        calls = [c for c in snapshot.contracts if c.contract_type == "CALL"]
        puts = [c for c in snapshot.contracts if c.contract_type == "PUT"]
        p.call_volume = sum(c.volume or 0 for c in calls)
        p.put_volume = sum(c.volume or 0 for c in puts)
        p.call_open_interest = sum(c.open_interest or 0 for c in calls)
        p.put_open_interest = sum(c.open_interest or 0 for c in puts)
        p.call_premium = sum(c.premium_volume or 0 for c in calls)
        p.put_premium = sum(c.premium_volume or 0 for c in puts)
        p.put_call_volume_ratio = round(p.put_volume / p.call_volume, 3) if p.call_volume else None
        p.put_call_premium_ratio = round(p.put_premium / p.call_premium, 3) if p.call_premium else None
        ratio = p.put_call_premium_ratio if p.put_call_premium_ratio is not None else (p.put_call_volume_ratio or 1.0)
        if bullish:
            if ratio <= 0.45: p.bias, p.flow_score = "CALL_BIASED", 82
            elif ratio <= 0.70: p.bias, p.flow_score = "CALL_LEAN", 68
            elif ratio >= 1.7: p.bias, p.flow_score = "PUT_BIASED", 22
            else: p.bias, p.flow_score = "NEUTRAL", 50
            eligible_types = ("CALL",)
        else:
            if ratio >= 1.7: p.bias, p.flow_score = "PUT_BIASED", 82
            elif ratio >= 1.4: p.bias, p.flow_score = "PUT_LEAN", 68
            elif ratio <= 0.45: p.bias, p.flow_score = "CALL_BIASED", 22
            else: p.bias, p.flow_score = "NEUTRAL", 50
            eligible_types = ("PUT",)

        p.confidence = min(95, 35 + min(60, int((p.call_volume + p.put_volume) / 750)))
        today = date.today()
        candidates=[]
        for c in snapshot.contracts:
            if c.contract_type not in eligible_types or not c.volume or not c.open_interest or c.expiration < today:
                continue
            mid=c.mid
            if not mid or mid <= 0:
                continue
            spread=c.spread_pct
            if spread is not None and spread > 15:
                continue
            dte=(c.expiration-today).days
            if dte < 7 or dte > 90:
                continue
            m=c.moneyness(price)
            if m is None:
                continue
            # Prefer slightly OTM, liquid contracts with sensible DTE and lower spread.
            target_m=-5.0 if bullish else 5.0  # put moneyness negative means OTM for puts? use absolute distance below
            distance=abs(m-5.0) if bullish else abs(m+5.0)
            dte_penalty=abs(dte-35)/35
            spread_penalty=(spread or 0)/15
            iv_penalty=max(0.0, (c.implied_volatility or 0)-1.0)
            score=100 - distance*5 - dte_penalty*15 - spread_penalty*20 - iv_penalty*10
            if c.open_interest >= 1000: score += 5
            if c.volume >= 1000: score += 5
            if c.delta is not None:
                # Prefer approximate delta 0.45-0.65 for directional contracts.
                score -= abs(abs(c.delta)-0.55)*30
            candidates.append((score,c))
        if candidates:
            candidates.sort(key=lambda x:x[0], reverse=True)
            p.contract_candidate=candidates[0][1]
            p.notes.append(f"Best contract score: {max(0,min(100,int(round(candidates[0][0]))))}/100")
        else:
            p.notes.append("No liquid contract met DTE/volume/OI/spread criteria")
        p.notes.append("Chain-derived bias; not execution or institutional-flow proof")
        return p
