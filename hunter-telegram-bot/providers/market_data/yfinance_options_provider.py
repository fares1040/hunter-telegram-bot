"""Free fallback option-chain provider using yfinance. Chain data only."""
import asyncio
from datetime import date, datetime
from typing import Optional
import yfinance as yf
from models.options import OptionContract, OptionsSnapshot
from providers.market_data.yfinance_concurrency import get_yfinance_semaphore


class YFinanceOptionsProvider:
    name = "yfinance_options"

    async def fetch_options(self, ticker: str, price: Optional[float] = None) -> OptionsSnapshot:
        async with get_yfinance_semaphore():
            try:
                stock = await asyncio.to_thread(yf.Ticker, ticker)
                expirations = await asyncio.to_thread(lambda: stock.options)
                if not expirations:
                    return OptionsSnapshot(ticker=ticker, underlying_price=price, source=self.name)
                # Use the nearest expiration with enough time to avoid same-day noise.
                exp = next((e for e in expirations if date.fromisoformat(e) >= date.today()), expirations[0])
                chain = await asyncio.to_thread(stock.option_chain, exp)
                contracts = []
                for typ, frame in (("CALL", chain.calls), ("PUT", chain.puts)):
                    for row in frame.itertuples(index=False):
                        contracts.append(OptionContract(
                            ticker=ticker,
                            contract_symbol=str(getattr(row, "contractSymbol", "")),
                            contract_type=typ,
                            strike=float(getattr(row, "strike", 0)),
                            expiration=date.fromisoformat(exp),
                            bid=float(getattr(row, "bid", 0) or 0),
                            ask=float(getattr(row, "ask", 0) or 0),
                            last=float(getattr(row, "lastPrice", 0) or 0),
                            volume=int(getattr(row, "volume", 0) or 0),
                            open_interest=int(getattr(row, "openInterest", 0) or 0),
                            implied_volatility=float(getattr(row, "impliedVolatility", 0) or 0),
                            delta=None, gamma=None, theta=None, vega=None,
                            source=self.name,
                        ))
                return OptionsSnapshot(ticker=ticker, underlying_price=price, contracts=contracts, source=self.name, timestamp=datetime.utcnow().isoformat())
            except Exception:
                return OptionsSnapshot(ticker=ticker, underlying_price=price, source=self.name)
