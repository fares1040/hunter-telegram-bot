"""Optional Polygon options snapshot provider. Requires POLYGON_API_KEY."""
import aiohttp
from datetime import date, datetime, timezone
from typing import Optional
from models.options import OptionContract, OptionsSnapshot

class PolygonOptionsProvider:
    name = "polygon_options"
    def __init__(self, api_key: str): self.api_key = api_key

    async def fetch_options(self, ticker: str, price: Optional[float] = None) -> OptionsSnapshot:
        url = f"https://api.polygon.io/v3/snapshot/options/{ticker}"
        params = {"apiKey": self.api_key, "limit": 250}
        contracts=[]
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=12)) as resp:
                    if resp.status != 200: return OptionsSnapshot(ticker=ticker, underlying_price=price, source=self.name)
                    payload = await resp.json()
            for item in payload.get("results", []):
                details=item.get("details", {}); day=item.get("day", {}); quote=item.get("last_quote", {})
                exp=details.get("expiration_date"); strike=details.get("strike_price")
                if not exp or strike is None: continue
                contracts.append(OptionContract(
                    ticker=ticker,
                    contract_symbol=details.get("ticker", ""),
                    contract_type=str(details.get("contract_type", "")).upper(),
                    strike=float(strike), expiration=date.fromisoformat(exp),
                    bid=quote.get("bid"), ask=quote.get("ask"), last=day.get("close"),
                    volume=int(day.get("volume") or 0), open_interest=int(item.get("open_interest") or 0),
                    implied_volatility=item.get("implied_volatility"), delta=(item.get("greeks") or {}).get("delta"), gamma=(item.get("greeks") or {}).get("gamma"), theta=(item.get("greeks") or {}).get("theta"), vega=(item.get("greeks") or {}).get("vega"), source=self.name,
                ))
            return OptionsSnapshot(ticker=ticker, underlying_price=price, contracts=contracts, source=self.name, timestamp=datetime.now(timezone.utc).isoformat())
        except Exception:
            return OptionsSnapshot(ticker=ticker, underlying_price=price, source=self.name)
