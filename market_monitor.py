#!/usr/bin/env python3
"""
Stockbee-style Market Monitor
=============================

Rebuilds Pradeep Bonde's (Stockbee) Market Monitor breadth table for US stocks
using free data (NASDAQ Trader symbol lists + yfinance).

Columns (one row per trading day, newest first):

  PRIMARY BREADTH
    Up 4%+ today        close >= +4% vs prior close, volume >= 100k and > yesterday's volume
    Down 4%+ today      close <= -4% vs prior close, same volume rules
    5-day ratio         sum(up 4%) / sum(down 4%) over the last 5 days
    10-day ratio        same over the last 10 days
    Up 25%+ quarter     close >= 25% above its lowest close of the last 65 days
    Down 25%+ quarter   close >= 25% below its highest close of the last 65 days

  SECONDARY BREADTH
    Up / Down 25%+ month   same idea, 20-day window
    Up / Down 50%+ month   50% move, 20-day window
    Up / Down 13%+ 34 days 13% move, 34-day window ("34/13 bull/bear")
    Universe            number of stocks that passed the liquidity filter
    T2108               % of the universe closing above its 40-day simple moving average
    S&P 500             index close

Market phase (Stockbee's primary rule): bullish when Up 25% quarter > Down 25% quarter.

Usage
-----
  pip install yfinance pandas numpy
  python market_monitor.py                 # first run downloads ~2y of data (10-20 min)
  python market_monitor.py                 # later runs only fetch the last few days
  python market_monitor.py --full          # force a full re-download (do weekly: fixes split adjustments)
  python market_monitor.py --rows 120      # show more history
  python market_monitor.py --tickers my_list.txt   # use your own universe (one ticker per line)
  python market_monitor.py --demo          # synthetic data, no internet - just to see the layout

Output: market_monitor.html (open in a browser) and market_monitor.csv
"""

from __future__ import annotations

import argparse
import io
import pickle
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")  # hide noisy pandas/yfinance deprecation warnings

# ----------------------------------------------------------------------------- settings
MIN_PRICE = 3.0            # ignore stocks below this close
MIN_AVG_VOLUME = 100_000   # 20-day average volume needed to be in the universe
MIN_DAY_VOLUME = 100_000   # volume needed on the day for a 4% move to count
HISTORY = "3y"             # download 3y so 2 full years can be shown after the 65-day warm-up
CHUNK = 400                # tickers per yfinance request
CACHE = Path("mm_cache.pkl")
TICKER_CACHE = Path("mm_tickers.txt")

# Colour thresholds (tweak to taste)
THRUST_4PCT = 300          # 300+ stocks up (or down) 4% in a day = notable thrust
RATIO_HI, RATIO_LO = 2.0, 0.5
EXTREME_50PCT = 20         # 20+ stocks moving 50% in a month is an extreme
T2108_LO, T2108_HI = 20, 70

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
JUNK_WORDS = ("warrant", "right", " unit", "units", "preferred", "depositary",
              "notes due", "debenture", "% ", "acquisition corp")


# ----------------------------------------------------------------------------- universe
def _fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "ignore")
        except Exception as e:
            last = e
            time.sleep(5 * (attempt + 1))
    raise last


def load_universe(path: str | None) -> list[str]:
    """Common stocks listed on NASDAQ / NYSE / NYSE American, ETFs and junk removed."""
    if path:
        tickers = [t.strip().upper() for t in Path(path).read_text().split() if t.strip()]
        return sorted(set(tickers))

    try:
        tickers = _download_universe()
        TICKER_CACHE.write_text("\n".join(tickers))
        return tickers
    except Exception as e:
        if TICKER_CACHE.exists():
            print(f"Couldn't fetch the ticker list ({e}); using the saved list instead.")
            return TICKER_CACHE.read_text().split()
        if CACHE.exists():
            print(f"Couldn't fetch the ticker list ({e}); using tickers from the saved prices.")
            return list(pickle.loads(CACHE.read_bytes())["close"].columns)
        raise


def _download_universe() -> list[str]:
    nas = pd.read_csv(io.StringIO(_fetch_text(NASDAQ_URL)), sep="|")
    nas = nas[(nas["Test Issue"] == "N") & (nas["ETF"] == "N") & (nas["Financial Status"] == "N")]
    nas = nas.rename(columns={"Symbol": "sym", "Security Name": "name"})

    oth = pd.read_csv(io.StringIO(_fetch_text(OTHER_URL)), sep="|")
    oth = oth[(oth["Test Issue"] == "N") & (oth["ETF"] == "N")]
    oth = oth.rename(columns={"ACT Symbol": "sym", "Security Name": "name"})

    df = pd.concat([nas[["sym", "name"]], oth[["sym", "name"]]]).dropna()
    name = df["name"].str.lower()
    df = df[~name.apply(lambda n: any(w in n for w in JUNK_WORDS))]
    df = df[~df["sym"].str.contains(r"[\$\.\^=]", regex=True)]      # preferreds / classes / oddities
    df = df[~df["sym"].str.startswith("File Creation")]
    return sorted(set(df["sym"].str.strip()))


# ----------------------------------------------------------------------------- prices
def _download(tickers: list[str], **kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf
    warnings.simplefilter("ignore")

    closes, vols = [], []
    for i in range(0, len(tickers), CHUNK):
        batch = tickers[i:i + CHUNK]
        print(f"  downloading {i + 1}-{i + len(batch)} of {len(tickers)} ...", flush=True)
        for attempt in range(3):
            try:
                d = yf.download(batch, auto_adjust=True, progress=False, threads=True,
                                group_by="column", **kw)
                break
            except Exception as e:  # network hiccup / rate limit
                print(f"    retry ({e})")
                time.sleep(5 * (attempt + 1))
        else:
            continue
        if d.empty:
            continue
        c, v = d["Close"], d["Volume"]
        if isinstance(c, pd.Series):          # single ticker edge case
            c, v = c.to_frame(batch[0]), v.to_frame(batch[0])
        closes.append(c)
        vols.append(v)
        time.sleep(1)
    close = pd.concat(closes, axis=1) if closes else pd.DataFrame()
    vol = pd.concat(vols, axis=1) if vols else pd.DataFrame()
    return close, vol


def get_prices(tickers: list[str], full: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import yfinance as yf
    warnings.simplefilter("ignore")

    cache = pickle.loads(CACHE.read_bytes()) if CACHE.exists() and not full else None

    if cache is None:
        print(f"Full download for {len(tickers)} tickers ...")
        close, vol = _download(tickers, period=HISTORY)
    else:
        close, vol = cache["close"], cache["vol"]
        start = (close.index.max() - timedelta(days=10)).strftime("%Y-%m-%d")
        new = [t for t in tickers if t not in close.columns]
        print(f"Updating from {start} ({len(tickers)} tickers, {len(new)} new) ...")
        c2, v2 = _download(tickers, start=start)
        close = c2.combine_first(close)     # fresh data wins
        vol = v2.combine_first(vol)
        if new:
            c3, v3 = _download(new, period=HISTORY)
            close, vol = close.combine_first(c3), vol.combine_first(v3)

    close = close.sort_index()
    vol = vol.reindex_like(close)
    CACHE.write_bytes(pickle.dumps({"close": close, "vol": vol}))

    idx = pd.DataFrame(index=close.index)
    for col, sym in (("spx", "^GSPC"), ("spy", "SPY")):
        d = yf.download(sym, period=HISTORY, auto_adjust=True, progress=False)["Close"]
        idx[col] = d.squeeze().reindex(close.index)
    return close, vol, idx


def demo_prices(n_stocks=3000, n_days=400, seed=7):
    """Random-walk market with regime changes, just to exercise the code and layout."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=datetime.today(), periods=n_days)
    regime = np.repeat(rng.normal(0, 0.004, n_days // 40 + 1), 40)[:n_days]
    mkt = regime + rng.normal(0, 0.009, n_days)
    beta = rng.uniform(0.5, 2.0, n_stocks)
    ret = mkt[:, None] * beta + rng.standard_t(3, (n_days, n_stocks)) * 0.018
    close = pd.DataFrame(20 * np.exp(np.cumsum(ret, axis=0)), idx,
                         [f"S{i}" for i in range(n_stocks)])
    vol = pd.DataFrame(rng.lognormal(12.5, 1.0, (n_days, n_stocks)) * (1 + 15 * np.abs(ret)),
                       idx, close.columns)
    spx = pd.Series(4000 * np.exp(np.cumsum(mkt)), idx)
    return close, vol, pd.DataFrame({"spx": spx, "spy": spx / 10})


# ----------------------------------------------------------------------------- the monitor
def above_below(price: pd.Series, avg: pd.Series) -> np.ndarray:
    return np.where(price.isna() | avg.isna(), "", np.where(price > avg, "Above", "Below"))


def compute(close: pd.DataFrame, vol: pd.DataFrame, idx: pd.DataFrame) -> pd.DataFrame:
    close = close.where(close > 0)
    liquid = (close >= MIN_PRICE) & (vol.rolling(20, min_periods=15).mean() >= MIN_AVG_VOLUME)

    chg = close / close.shift(1) - 1
    vol_ok = (vol >= MIN_DAY_VOLUME) & (vol > vol.shift(1))
    up4 = ((chg >= 0.04) & vol_ok & liquid).sum(axis=1)
    dn4 = ((chg <= -0.04) & vol_ok & liquid).sum(axis=1)

    def ratio(n):
        u, d = up4.rolling(n).sum(), dn4.rolling(n).sum()
        return (u / d.replace(0, np.nan)).round(2)

    def up_from_low(n, pct):
        return ((close / close.rolling(n, min_periods=n).min() - 1 >= pct) & liquid).sum(axis=1)

    def down_from_high(n, pct):
        return ((close / close.rolling(n, min_periods=n).max() - 1 <= -pct) & liquid).sum(axis=1)

    sma40 = close.rolling(40, min_periods=40).mean()
    above40 = ((close > sma40) & liquid & sma40.notna()).sum(axis=1)
    universe = liquid.sum(axis=1)

    mm = pd.DataFrame({
        "up4": up4,
        "dn4": dn4,
        "r5": ratio(5),
        "r10": ratio(10),
        "up25q": up_from_low(65, 0.25),
        "dn25q": down_from_high(65, 0.25),
        "up25m": up_from_low(20, 0.25),
        "dn25m": down_from_high(20, 0.25),
        "up50m": up_from_low(20, 0.50),
        "dn50m": down_from_high(20, 0.50),
        "up13_34": up_from_low(34, 0.13),
        "dn13_34": down_from_high(34, 0.13),
        "universe": universe,
        "t2108": (100 * above40 / universe.replace(0, np.nan)).round(1),
        "spx": idx["spx"].round(2),
        "spy_sma5": above_below(idx["spy"], idx["spy"].rolling(5).mean()),
        "spy_ema21": above_below(idx["spy"], idx["spy"].ewm(span=21, adjust=False).mean()),
        "spy_sma50": above_below(idx["spy"], idx["spy"].rolling(50).mean()),
    })
    mm["pbr"] = (mm["up25q"] / mm["dn25q"].replace(0, np.nan)).round(2)   # primary breadth ratio
    mm["phase"] = mm.apply(phase_of, axis=1)
    # Risk On without a 300+ "up 4%" thrust day = Risk On, but cautious
    mm["cautious"] = (mm["phase"] == "Risk On") & (mm["up4"] < THRUST_4PCT)
    # need 65+ days of history before the quarter columns mean anything
    return mm.iloc[66:].dropna(subset=["r10"])


# ----------------------------------------------------------------------------- HTML
COLS = [  # key, header, group
    ("phase", "Market phase", "d"),
    ("up4", "Number of stocks up 4% plus today", "p"),
    ("dn4", "Number of stocks down 4% plus today", "p"),
    ("r5", "5 day ratio", "p"),
    ("r10", "10 day ratio", "p"),
    ("spy_sma5", "SPY vs 5-day SMA", "p"),
    ("up25q", "Number of stocks up 25% plus in a quarter", "p"),
    ("dn25q", "Number of stocks down 25% + in a quarter", "p"),
    ("up25m", "Number of stocks up 25% + in a month", "s"),
    ("dn25m", "Number of stocks down 25% + in a month", "s"),
    ("up50m", "Number of stocks up 50% + in a month", "s"),
    ("dn50m", "Number of stocks down 50% + in a month", "s"),
    ("up13_34", "Number of stocks up 13% + in 34 days", "s"),
    ("dn13_34", "Number of stocks down 13% + in 34 days", "s"),
    ("spy_ema21", "SPY vs 21-day EMA", "s"),
    ("spy_sma50", "SPY vs 50-day SMA", "s"),
    ("universe", "Stock universe (liquid)", "o"),
    ("t2108", "T2108 (% above 40-day MA)", "o"),
    ("spx", "S&P 500", "o"),
]


PRIMARY = ("up4", "dn4", "r5", "r10", "spy_sma5", "up25q", "dn25q")
GREENS, REDS = {"gd", "gb"}, {"rd", "pk"}


def phase_of(row: pd.Series) -> str:
    """Risk On / Risk Off by whether the primary breadth cells are mostly green or mostly red."""
    cls = [cell_class(k, row) for k in PRIMARY]
    g = sum(c in GREENS for c in cls)
    r = sum(c in REDS for c in cls)
    return "Risk On" if g > r else "Risk Off" if r > g else "Neutral"


def cell_class(key: str, row: pd.Series) -> str:
    """Stockbee colour scheme:
    gd = dark green, gb = bright green, pk = light pink, rd = red, "" = no colour."""
    v = row[key]
    if key == "phase":
        return {"Risk On": "gb", "Risk Off": "rd"}.get(v, "nt")
    if pd.isna(v):
        return ""
    up, dn = row["up4"], row["dn4"]
    if key == "up4":                       # up side winning: green (dark if a 300+ thrust); losing: pink
        return ("gd" if v >= THRUST_4PCT else "gb") if up > dn else "pk" if up < dn else ""
    if key == "dn4":                       # down side losing: green; winning: pink (red if 300+)
        return "gb" if dn < up else ("rd" if v >= THRUST_4PCT else "pk") if dn > up else ""
    if key in ("r5", "r10"):
        return "rd" if v < RATIO_LO else "gb" if v >= RATIO_HI else ""
    pairs = {"up25q": "dn25q", "up25m": "dn25m", "up13_34": "dn13_34"}
    rev = {d: u for u, d in pairs.items()}
    if key in pairs or key in rev:         # both cells green when ups lead, both red when downs lead
        u = row[key] if key in pairs else row[rev[key]]
        d = row[pairs[key]] if key in pairs else row[key]
        return "gb" if u > d else "rd" if d > u else ""
    if key == "up50m":                     # 20+ stocks up 50% in a month = froth (red)
        return "rd" if v >= EXTREME_50PCT else ""
    if key == "dn50m":                     # 20+ stocks down 50% in a month = washout (contrarian green)
        return "gb" if v >= EXTREME_50PCT else ""
    if key in ("spy_sma5", "spy_ema21", "spy_sma50"):
        return {"Above": "gb", "Below": "pk"}.get(v, "")
    if key == "t2108":
        return "gb" if v <= T2108_LO else "rd" if v >= T2108_HI else ""
    return ""


CAUTION_TAG = '<span class="tag">Cautious</span>'


def state_html(row) -> str:
    return row["phase"] + (CAUTION_TAG if row.get("cautious", False) else "")


def fmt(key, v):
    if key in ("phase", "spy_sma5", "spy_ema21", "spy_sma50"):
        return v
    if pd.isna(v):
        return "–"
    if key in ("r5", "r10", "pbr"):
        return f"{v:.2f}"
    if key == "t2108":
        return f"{v:.1f}"
    if key == "spx":
        return f"{v:,.2f}"
    return f"{int(v):,}"


def to_html(mm: pd.DataFrame, rows: int, demo: bool) -> str:
    view = mm.iloc[::-1].head(rows)
    last = mm.iloc[-1]
    phase = last["phase"].upper()
    box_cls = {"RISK ON": "bull", "RISK OFF": "bear"}.get(phase, "neu")

    head1 = ('<tr><th rowspan="2" class="date d">Date</th>'
             '<th rowspan="2" class="d">Market state</th>'
             '<th colspan="7" class="grp p">Primary Breadth Indicators</th>'
             '<th colspan="8" class="grp s">Secondary Breadth Indicators</th>'
             '<th colspan="3" class="grp o"></th></tr>')
    head2 = "<tr>" + "".join(f'<th class="{g}">{h}</th>' for k, h, g in COLS if k != "phase") + "</tr>"

    body = []
    prev_month = None
    for d, row in view.iterrows():
        sep = ' class="msep"' if prev_month and d.month != prev_month else ""
        prev_month = d.month
        tds = "".join(f'<td class="{cell_class(k, row)}{" phase" if k in ("phase", "spy_sma5", "spy_ema21", "spy_sma50") else ""}">{state_html(row) if k == "phase" else fmt(k, row[k])}</td>'
                      for k, _, _ in COLS)
        body.append(f"<tr{sep}><td class='date'>{d.month}/{d.day}/{d.year}</td>{tds}</tr>")

    note = "<p class='demo'>DEMO DATA – synthetic prices, not the real market.</p>" if demo else ""
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Market Monitor – {mm.index[-1]:%d %b %Y}</title>
<style>
 body{{font-family:Arial,Helvetica,sans-serif;margin:16px;color:#222;background:#fff}}
 h1{{font-size:20px;margin:0 0 4px}}
 .sum{{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0 14px}}
 .box{{border:1px solid #ccc;border-radius:6px;padding:8px 12px;min-width:120px}}
 .box b{{display:block;font-size:20px}} .box span{{font-size:11px;color:#666;text-transform:uppercase}}
 .bull b{{color:#137333}} .bear b{{color:#c5221f}} .neu b{{color:#b06000}}
 .wrap{{overflow:auto;max-height:80vh;border:1px solid #bbb}}
 table{{border-collapse:separate;border-spacing:0;font-size:12px;white-space:nowrap}}
 th,td{{border:0;border-right:1px solid #d0d0d0;border-bottom:1px solid #d0d0d0;padding:4px 7px;text-align:center}}
 thead{{position:sticky;top:0;z-index:3}}
 thead th{{font-weight:bold;white-space:normal;max-width:84px;
   vertical-align:middle;font-size:12px;color:#222}}
 th.grp{{max-width:none;font-size:16px;padding:6px}}
 th.d,th.o{{background:#fdf3dc}} th.p{{background:#b8f1f5}} th.s{{background:#fff7a8}}
 th.date,td.date{{position:sticky;left:0;background:#fff;z-index:1;font-weight:bold}}
 thead th.date{{z-index:3;background:#fdf3dc}}
 td{{font-size:12px;color:#222}}
 td.gd{{background:#93c47d}} td.gb{{background:#d9ead3}}
 td.pk{{background:#fce8e6}} td.rd{{background:#f2b8b5}}
 td.nt{{background:#fff2cc}}
 td.phase{{font-weight:bold}}
 td.phase:nth-child(2){{text-align:left;padding-left:12px;min-width:128px;white-space:nowrap}}
 .tag{{display:inline-block;margin-left:6px;padding:1px 6px;border-radius:9px;background:#fbe3b4;
   color:#8a5300;font-size:10px;font-weight:bold;text-transform:uppercase;letter-spacing:.3px;
   vertical-align:middle;border:1px solid #f0c27a}}
 .box .tag{{font-size:11px;margin-left:8px}}
 tr.msep td{{box-shadow:inset 0 2px 0 #666}}
 tbody tr:hover td{{outline:1px solid #999}}
 .legend{{font-size:11px;color:#555;margin-top:10px;line-height:1.6}}
 .legend i{{display:inline-block;width:12px;height:12px;vertical-align:middle;margin:0 3px 0 10px}}
 .demo{{color:#c5221f;font-weight:bold}}
</style></head><body>
<h1>Market Monitor</h1>
<div>Last update: {mm.index[-1]:%A %d %B %Y} · generated {datetime.now():%Y-%m-%d %H:%M}</div>{note}
<div class="sum">
 <div class="box {box_cls}"><span>Market state</span><b>{phase}{CAUTION_TAG if last["cautious"] else ""}</b></div>
 <div class="box"><span>Primary breadth ratio</span><b>{fmt('pbr', last['pbr'])}</b></div>
 <div class="box"><span>10-day ratio</span><b>{fmt('r10', last['r10'])}</b></div>
 <div class="box"><span>T2108</span><b>{fmt('t2108', last['t2108'])}</b></div>
 <div class="box"><span>Up / down 4% today</span><b>{int(last['up4'])} / {int(last['dn4'])}</b></div>
</div>
<div class="wrap"><table><thead>{head1}{head2}</thead><tbody>
{''.join(body)}
</tbody></table></div>
<div class="legend">
 <b>Colours (Stockbee style):</b>
 <i style="background:#93c47d"></i>300+ stocks up 4% and beating the down count
 <i style="background:#d9ead3"></i>bullish side leading
 <i style="background:#fce8e6"></i>down side leading (mild)
 <i style="background:#f2b8b5"></i>bearish (300+ down 4%, ratio &lt; 0.5, downs lead on 25%/13% moves, 20+ up 50% in a month)<br>
 Universe: US common stocks, close ≥ ${MIN_PRICE:g}, 20-day avg volume ≥ {MIN_AVG_VOLUME:,}.
 4% moves also need volume ≥ {MIN_DAY_VOLUME:,} and above the prior day.
 Quarter = 65 trading days, month = 20. Up X% = close vs lowest close in the window; down X% = close vs highest close.
 Primary breadth ratio = up 25% quarter ÷ down 25% quarter.<br>
 <b>Market state:</b> count the coloured cells in the seven Primary Breadth columns.
 More green than red = Risk On, more red than green = Risk Off, equal = Neutral.
 <span class="tag" style="margin-left:0">Cautious</span> = Risk On, but fewer than {THRUST_4PCT} stocks were up 4%+ that day.
</div>
</body></html>"""


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Stockbee-style Market Monitor")
    ap.add_argument("--rows", type=int, default=504, help="days to show in the table (504 = ~2 years)")
    ap.add_argument("--full", action="store_true", help="ignore cache, re-download everything")
    ap.add_argument("--tickers", help="text file of tickers to use instead of all US stocks")
    ap.add_argument("--demo", action="store_true", help="synthetic data, no internet")
    ap.add_argument("--out", default="market_monitor", help="output file name (no extension)")
    a = ap.parse_args()

    if a.demo:
        close, vol, idx = demo_prices()
    else:
        tickers = load_universe(a.tickers)
        print(f"Universe: {len(tickers)} tickers")
        close, vol, idx = get_prices(tickers, a.full)

    mm = compute(close, vol, idx)
    if mm.empty:
        sys.exit("Not enough history to compute the monitor.")

    mm.iloc[::-1].to_csv(f"{a.out}.csv", index_label="date")
    Path(f"{a.out}.html").write_text(to_html(mm, a.rows, a.demo), encoding="utf-8")
    last = mm.iloc[-1]
    print(f"\n{mm.index[-1]:%Y-%m-%d}  up4={int(last.up4)} dn4={int(last.dn4)} "
          f"r5={last.r5} r10={last.r10} up25q={int(last.up25q)} dn25q={int(last.dn25q)} "
          f"T2108={last.t2108}")
    print(f"Wrote {a.out}.html and {a.out}.csv")


if __name__ == "__main__":
    main()
