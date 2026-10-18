# Phase 4 Plan: Bloomberg integration (written, not implemented)

Status: plan only. Nothing in this document has been built. Phases 1-3 stay as they are until this plan is approved and run on a machine with a live Bloomberg terminal.

Spec: `Build_Plan.md` Section 13 (Phase 4), Sections 3.4 and 14 (open item 3), plus items deferred from Phases 1-3.

## A. Goal

Run the notebook with Bloomberg as the primary source wherever Build_Plan names one, verify every ticker on a live terminal, and bring the three Bloomberg-only factors (A6, F6, G6) into the composite. Then re-test, with the same selection-honest method used in Phase 3, whether the extra data improves the signal (the 10y in particular). If it does not, the live configuration stays as it is.

## B. Scope

| # | Item | Source of the requirement |
|---|---|---|
| 1 | Install xbbg and blpapi on the terminal machine; verify the Desktop API connection | Build_Plan 3.4, Phase 1 decision 7 |
| 2 | Verify every ticker in `tickers_bloomberg.yaml` on a live terminal and set `verified: true` | Build_Plan 13 Phase 4 |
| 3 | Bloomberg-primary factors populate from Bloomberg: D2 (1y OIS minus fed funds), F1 fallback (ACM term premium), F4 fallback (10y TIPS real yield), H1 (Bund 10y), H2 (JGB 10y), H4 (HY OAS) | Build_Plan 5 |
| 4 | Bloomberg-only factors enter the composite: A6 (Citi economic surprise), F6 (10y swap spread), G6 (MOVE) | Build_Plan 5 |
| 5 | Regime model inputs switch to Bloomberg: SPX for the equity-bond correlation (currently NASDAQCOM), HY OAS for the credit-spread change (currently Moody's Baa minus 10y) | Phase 3 decision 1 |
| 6 | Fed dots versus market pricing as a new D-theme factor (FOMC Summary of Economic Projections median versus OIS) | Build_Plan 14, open item 3 |
| 7 | Overlap check: Bloomberg and FRED agree on shared series | Build_Plan 13 Phase 4 |
| 8 | Re-run the nested calibration with the full factor set and report whether anything changed | Phase 3 method |

Out of scope: Bloomberg macro (ECO) data for backtesting. Bloomberg revises macro history, so ALFRED stays the point-in-time source (Build_Plan 3.4). Bloomberg macro may be used in live mode only as a freshness check.

## C. Decisions to confirm before starting

1. **Ticker choices to verify on the terminal** (all currently `verified: false`):

| Factor | Current ticker | What to confirm |
|---|---|---|
| A6 | `CESIUSD Index` | Returns daily history back to at least 2003 |
| D2 | `USOSFR1 Curncy` minus `FEDL01 Index` | `USOSFR1` is the 1y SOFR OIS; SOFR OIS history starts around 2018, so decide whether to splice to 1y fed funds OIS before that, or keep the FRED fallback (1y bill minus fed funds) for history and Bloomberg only live |
| F1 | `ACMTP10 Index` | ACM 10y term premium; used only as a fallback to Kim-Wright |
| F4 | `USGGT10Y Index` | Generic 10y TIPS real yield; compare with FRED DFII10 |
| F6 | `USSP10 Curncy` | Which swap curve it references after the LIBOR transition (SOFR versus LIBOR), and whether a continuous history exists; choose a SOFR-based spread ticker if not |
| G6 | `MOVE Index` | Daily history back to at least 2003 |
| H1 | `GDBR10 Index` | Generic Bund 10y |
| H2 | `GJGB10 Index` | Generic JGB 10y |
| H4 | `LF98OAS Index` | Bloomberg US Corporate High Yield OAS, in bp |
| Regime | `SPX Index` | Replaces NASDAQCOM |
| Overlap | `USGG2YR Index`, `USGG10YR Index` | For the 2bp agreement check against FRED DGS2 / DGS10 |
| D (new) | to be identified | FOMC SEP median policy rate for the coming year-ends; identify the correct ticker on the terminal before building |

2. **History length.** Every factor needs 5 years of history before its z-score counts. A Bloomberg series that starts later than its FRED fallback shortens that factor's history. Rule to confirm: use Bloomberg for history only where it starts no later than the fallback; otherwise splice (fallback history, Bloomberg from its start date) and record the splice date in `panel_meta.json`.
3. **Publication timing.** Bloomberg market data is end-of-day and observable on the day, so release date = observation date, as for FRED daily market series.
4. **Fed dots release dates.** SEP projections are published at four FOMC meetings a year; the factor enters the panel on the meeting date, never earlier.

## D. Tasks and success criteria

### Task 1: Environment on the terminal machine
- [ ] Install `blpapi` from the Bloomberg index and `xbbg`; pin versions in `requirements.txt` (the commented block becomes live for this machine)
- [ ] Criterion: a one-line `blp.bdh("USGG10YR Index", "PX_LAST", ...)` returns data; the existing test suite still passes with Bloomberg present

### Task 2: Ticker verification
- [ ] Script `python -m ust_rv.providers.verify_bloomberg` pulls each ticker, reports first date, last date, row count and a sample of values, and writes nothing to config by itself
- [ ] Jeff reviews the report; tickers that pass are set to `verified: true` by hand
- [ ] Criterion: every ticker in `tickers_bloomberg.yaml` is either verified or replaced with a verified alternative; the notebook's unverified-ticker warning is empty

### Task 3: Bloomberg as primary source
- [ ] Provider adapter tested against the live terminal (pull, cache, stale fallback when the session drops)
- [ ] Splice rule from C.2 implemented and shown in the factor library table
- [ ] Criterion: all Bloomberg-primary factors populate from Bloomberg with no stale flags; A6, F6, G6 show status `ok`

### Task 4: Overlap check
- [ ] Compare `USGG10YR Index` with FRED DGS10 and `USGG2YR Index` with DGS2 over the last 5 years
- [ ] Criterion: Build_Plan 13 Phase 4 acceptance: scorecard values for FRED-overlap series agree within 2bp (report the share of days within 2bp and the largest gap with its date)

### Task 5: Regime inputs
- [ ] Regime features use SPX and HY OAS when Bloomberg is available, NASDAQCOM and Baa-10y otherwise; the source used is recorded
- [ ] HMM fit cache keys change automatically because the training data changes
- [ ] Criterion: regime tests pass; the regime panel shows which inputs were used

### Task 6: Fed dots factor
- [ ] Add factor D6: SEP median policy rate for the next year-end minus the 1y OIS (dots above market = the Fed signals more tightening than priced). Sign and weights to be proposed with evidence before it enters the composite
- [ ] Criterion: point-in-time test proves each SEP value appears only from its meeting date

### Task 7: Re-test with the full factor set
- [ ] Re-run `python -m ust_rv.backtest.calibrate` (nested selection, same rules and parsimony margin as Phase 3)
- [ ] Write `runs/phase4_results.md`: nested results with and without the Bloomberg factors, per tenor, with confidence intervals
- [ ] Criterion: the live configuration changes only if the nested out-of-sample evidence supports it; otherwise the memo says the Bloomberg factors did not improve the signal

### Task 8: Phase 4 acceptance
- [ ] Full `pytest` green on the terminal machine (including Bloomberg-marked live tests) and on a machine without Bloomberg (Bloomberg factors fall back or show unavailable, as today)
- [ ] Notebook runs clean on both machines

## E. Risks

- Bloomberg history for some series is shorter than FRED's; splicing changes z-score history and must be visible, not silent.
- Swap spread definitions changed with the LIBOR transition; a spliced series can contain a level break.
- Adding factors after seeing Phase 1-3 results is itself a form of selection. Task 7 judges them with the same nested out-of-sample test, so they must earn their place.
- The Desktop API only works on the terminal machine. Every run elsewhere must keep working on fallbacks, as it does now.
