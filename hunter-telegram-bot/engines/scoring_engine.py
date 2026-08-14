"""Hunter composite score with normalized weights and explicit context gates."""

class CompositeScoringEngine:
    # Sum = 1.00. Trap risk is handled as a gate/multiplier, not as a hidden
    # subtraction that makes the displayed score inconsistent.
    WEIGHTS = {
        "news_quality": 0.10,
        "news_impact": 0.15,
        "reaction": 0.18,
        "liquidity": 0.12,
        "technical": 0.14,
        "options": 0.10,
        "market_regime": 0.08,
        "sector_strength": 0.05,
        "risk": 0.08,
    }

    def score(self, *, news_quality: int, news_impact: int, reaction: int, liquidity: int,
              technical: int, options: int, risk: int, market_regime: int = 50,
              sector_strength: int = 50, trap_risk: int = 0) -> int:
        raw = sum(values * self.WEIGHTS[name] for name, values in {
            "news_quality": news_quality, "news_impact": news_impact,
            "reaction": reaction, "liquidity": liquidity, "technical": technical,
            "options": options, "market_regime": market_regime,
            "sector_strength": sector_strength, "risk": risk,
        }.items())
        multiplier = 1.0
        if trap_risk >= 75:
            multiplier = 0.0
        elif trap_risk >= 60:
            multiplier = 0.80
        elif trap_risk >= 40:
            multiplier = 0.92
        return max(0, min(100, int(round(raw * multiplier))))
