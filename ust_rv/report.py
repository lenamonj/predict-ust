"""Notebook rendering: header, scorecard styling, charts and appendix tables."""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ust_rv.config import factors as load_factors
from ust_rv.config import settings, thresholds
from ust_rv.runner import RunResult
from ust_rv.scoring.scorecard import TENORS

RV_TEXT = {1: "+1 BUY", 0: "0 NEUTRAL", -1: "-1 SELL"}
RV_COLOR = {1: "#1a7f37", 0: "#57606a", -1: "#cf222e"}
TENOR_COLOR = {"2y": "#0969da", "10y": "#8250df"}
LAYOUT = dict(template="plotly_white", legend=dict(orientation="v", x=1.02, y=1, xanchor="left"), margin=dict(r=160))


def _asof(result: RunResult) -> str:
    return f"{result.panel.market_date:%Y-%m-%d}"


# ---------- section 0: header ----------

def header_html(result: RunResult) -> str:
    sig, card = result.signal, result.card
    status = card["status"].value_counts().to_dict()
    weighted = card[(card["w_2y"] > 0) | (card["w_10y"] > 0)]
    not_fresh = weighted[~weighted["fresh"] & weighted["status"].isin(["ok", "stale"])]

    tiles = ""
    for tenor in TENORS:
        s = sig[tenor]
        tiles += (
            f'<div style="display:inline-block;margin-right:40px;vertical-align:top">'
            f'<div style="font-size:14px;color:#57606a">RV {tenor}</div>'
            f'<div style="font-size:40px;font-weight:700;color:{RV_COLOR[s["rv"]]}">{RV_TEXT[s["rv"]]}</div>'
            f'<div>composite z <b>{s["composite"]:+.2f}</b></div>'
            f'<div>next: {s["next_threshold"]}, needs {s["distance"]:+.2f}</div></div>'
        )

    if result.prior is None:
        change = "<p>No prior snapshot.</p>"
    else:
        p = result.prior
        change = (
            f'<p><b>Since last run ({pd.Timestamp(p["run_date"]):%Y-%m-%d}):</b> '
            f'RV 2y {int(p["rv_2y"]):+d} &rarr; {sig["2y"]["rv"]:+d} (composite {p["composite_2y"]:+.2f} &rarr; {sig["2y"]["composite"]:+.2f}); '
            f'RV 10y {int(p["rv_10y"]):+d} &rarr; {sig["10y"]["rv"]:+d} (composite {p["composite_10y"]:+.2f} &rarr; {sig["10y"]["composite"]:+.2f})</p>'
        )
    if not result.movers.empty:
        change += "<p>z moves &gt; 0.5: " + ", ".join(f"{i} {r.factor} ({r.dz:+.2f})" for i, r in result.movers.iterrows()) + "</p>"

    warn = ""
    errors = card[card["status"] == "error"]
    if not errors.empty:
        warn += (
            '<div style="border-left:4px solid #cf222e;padding:6px 10px;background:#ffebe9;margin:8px 0">'
            "<b>Data errors (factor excluded):</b> "
            + html.escape("; ".join(f"{i} {'; '.join(result.panel.meta[i]['errors'])}" for i in errors.index)) + "</div>"
        )
    stale_txt = ", ".join(f"{k}: {v}" for k, v in sorted(status.items()))
    late = ", ".join(f"{i} ({int(r.age_days)}d)" for i, r in not_fresh.iterrows()) or "none"
    return (
        f'<h2 style="margin-bottom:4px">UST 2y / 10y RV Score</h2>'
        f'<div style="color:#57606a">Run {result.run_ts:%Y-%m-%d %H:%M} | market data through {_asof(result)} | curve read: <b>{sig["curve"]}</b></div>'
        f"<div style='margin:12px 0'>{tiles}</div>{change}{warn}"
        f"<p><b>Data status:</b> {stale_txt}. <b>Older than release lag:</b> {late}</p>"
    )


# ---------- section 1: master scorecard ----------

def scorecard_table(result: RunResult) -> pd.DataFrame:
    """Factor rows sorted by theme then |contribution|, with a subtotal row per theme."""
    card = result.card.copy()
    card["_abs"] = card[["contrib_2y", "contrib_10y"]].abs().max(axis=1)
    themes = result.themes
    blocks = []
    for theme in sorted(card["theme"].unique()):
        rows = card[card["theme"] == theme].sort_values("_abs", ascending=False)
        # subtotal: contributions sum to the theme pressure for each tenor
        sub = pd.DataFrame(
            [{
                "factor": f"{theme.upper()} theme", "theme": theme,
                **{f"contrib_{t}": themes[f"pressure_{t}"].get(theme, float("nan")) for t in TENORS},
            }],
            index=[theme],
        )
        blocks += [rows, sub]
    out = pd.concat(blocks)
    cols = ["factor", "value", "as_of", "age_days", "z", "z_chg_1m", "pressure", "w_2y", "w_10y",
            "contrib_2y", "contrib_10y", "sign_2y", "sign_10y", "source", "status"]
    return out[cols]


def style_scorecard(table: pd.DataFrame):
    def pressure_color(v):
        if pd.isna(v):
            return ""
        alpha = min(abs(v) / 2.0, 1.0) * 0.6
        rgb = "207,34,46" if v > 0 else "26,127,55"  # red = pushing yields higher, green = lower
        return f"background-color: rgba({rgb},{alpha:.2f})"

    def subtotal(row):
        bold = "font-weight:700;border-top:1px solid #8c959f;background-color:#f6f8fa"
        return [bold if str(row["factor"]).endswith(" theme") else "" for _ in row]

    def flag(v):
        return {"WARN": "color:#bf8700;font-weight:700", "unavailable": "color:#8c959f", "error": "color:#cf222e;font-weight:700",
                "stale": "color:#bf8700"}.get(v, "")

    fmt = {"value": "{:.2f}", "z": "{:+.2f}", "z_chg_1m": "{:+.2f}", "pressure": "{:+.2f}", "w_2y": "{:.2f}", "w_10y": "{:.2f}",
           "contrib_2y": "{:+.2f}", "contrib_10y": "{:+.2f}", "age_days": "{:.0f}", "as_of": lambda d: "" if pd.isna(d) else f"{d:%Y-%m-%d}"}
    table = table.rename_axis("ID")
    return (
        table.style.format(fmt, na_rep="")
        .apply(subtotal, axis=1)
        .map(pressure_color, subset=["pressure", "contrib_2y", "contrib_10y"])
        .map(flag, subset=["sign_2y", "sign_10y", "status"])
        .set_properties(subset=["as_of"], **{"white-space": "nowrap"})
        .relabel_index([SCORECARD_HEADERS[c] for c in table.columns], axis="columns")
        .set_table_styles([{"selector": "th.col_heading", "props": "white-space: normal; max-width: 110px; vertical-align: bottom;"}])
    )


SCORECARD_HEADERS = {
    "factor": "Factor", "value": "Today's value", "as_of": "Data released", "age_days": "Days since release",
    "z": "Z-score vs 10y history", "z_chg_1m": "Z-score change, last month", "pressure": "Pressure on yields (red = higher)",
    "w_2y": "Weight in theme, 2y", "w_10y": "Weight in theme, 10y", "contrib_2y": "Share of theme push, 2y",
    "contrib_10y": "Share of theme push, 10y", "sign_2y": "Sign check, 2y", "sign_10y": "Sign check, 10y",
    "source": "Data source", "status": "Data status",
}

THEME_HEADERS = {
    "pressure": "Average factor pressure", "z": "Theme z-score", "W": "Theme weight", "contrib": "Share of composite",
}


def theme_numbers(result: RunResult) -> pd.DataFrame:
    """The theme table with readable headings, grouped by tenor."""
    t = result.themes
    out = pd.DataFrame(index=t.index.rename("Theme"))
    for tenor in TENORS:
        for key, label in THEME_HEADERS.items():
            out[f"{label}, {tenor}"] = t[f"{key}_{tenor}"]
    return out.round(2)


# ---------- section 2: themes ----------


# ---------- section 3: composite and decision ----------

def fig_yield_signal(result: RunResult, tenor: str) -> go.Figure:
    """The actual yield since the first signal, shaded by the call in force: green = +1 (buy), red = -1 (sell)."""
    rv = result.scores.rv[tenor].dropna()
    y = result.panel.yields[{"2y": "y2", "10y": "y10"}[tenor]].loc[rv.index[0]:]
    fig = go.Figure()
    runs = (rv != rv.shift()).cumsum()
    for _, seg in rv.groupby(runs):
        v = int(seg.iloc[0])
        if v != 0:
            fig.add_vrect(x0=seg.index[0], x1=seg.index[-1], fillcolor=RV_COLOR[v], opacity=0.18, line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=y.index, y=y, name=f"{tenor} Treasury yield", line=dict(color="#24292f", width=1.4)))
    share = rv.value_counts(normalize=True)
    for v, name in ((1, "RV +1 (buy): expect yield lower"), (-1, "RV -1 (sell): expect yield higher")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color=RV_COLOR[v], symbol="square", size=12, opacity=0.5),
                                 name=f"{name}, {share.get(v, 0):.0%} of days"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="white", symbol="square", size=12, line=dict(color="#8c959f", width=1)),
                             name=f"neutral (0), {share.get(0, 0):.0%} of days"))
    fig.update_layout(title=f"{tenor} Treasury yield with the model's call shaded, {rv.index[0]:%Y} to date, as of {_asof(result)}",
                      yaxis_title=f"{tenor} yield, %", height=480, **LAYOUT)
    return fig


def fig_composite(result: RunResult, tenor: str) -> go.Figure:
    cfg = settings()
    start = cfg["backtest_start"]
    comp = result.scores.composite[tenor].loc[start:]
    rv = result.scores.rv[tenor].loc[start:]
    enter, exit_ = thresholds(tenor)

    fig = go.Figure()
    # shade signal states: green = buy (+1), red = sell (-1)
    state = rv.fillna(0)
    runs = (state != state.shift()).cumsum()
    for _, seg in state.groupby(runs):
        v = seg.iloc[0]
        if v != 0:
            fig.add_vrect(x0=seg.index[0], x1=seg.index[-1], fillcolor=RV_COLOR[int(v)], opacity=0.12, line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=comp.index, y=comp, name="composite z (lhs)", line=dict(color=TENOR_COLOR[tenor])))
    ycol = {"2y": "y2", "10y": "y10"}[tenor]
    yld = result.panel.yields[ycol].loc[start:]
    fig.add_trace(go.Scatter(x=yld.index, y=yld, name=f"{tenor} yield, % (rhs)", yaxis="y2", line=dict(color="#24292f", width=1.2)))
    for y, name, dash in [(enter, f"sell entry +{enter}", "dash"), (-enter, f"buy entry -{enter}", "dash"),
                          (exit_, f"exit band +/-{exit_}", "dot"), (-exit_, None, "dot")]:
        fig.add_hline(y=y, line_dash=dash, line_color="#57606a", line_width=1)
        if name:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(dash=dash, color="#57606a"), name=name))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color=RV_COLOR[1], symbol="square", opacity=0.4), name="RV +1 (buy)"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color=RV_COLOR[-1], symbol="square", opacity=0.4), name="RV -1 (sell)"))
    fig.update_layout(title=f"{tenor} composite, signal and actual {tenor} yield since {start[:4]}, as of {_asof(result)}",
                      yaxis_title="composite z (+ = yields higher)",
                      yaxis2=dict(title=f"{tenor} yield, %", overlaying="y", side="right", showgrid=False, tickmode="auto", tickformat=".1f"),
                      **LAYOUT)
    fig.update_layout(legend_x=1.08)
    return fig


def fig_band_position(result: RunResult) -> go.Figure:
    """Today's composite on the decision scale, with all five zones labelled and the move over the last 3 months."""
    from plotly.subplots import make_subplots

    tenors = list(TENORS)
    comp_all = {t: result.scores.composite[t].dropna() for t in tenors}
    span = max(1.5, *(abs(v) for t in tenors for v in (comp_all[t].iloc[-1], comp_all[t].iloc[-1 - BREAKDOWN_LOOKBACK_DAYS])))
    lim = min(3.0, round(span + 0.3, 1))
    fig = make_subplots(rows=len(tenors), cols=1, vertical_spacing=0.28, subplot_titles=[" "] * len(tenors))
    for row, tenor in enumerate(tenors, start=1):
        enter, exit_ = thresholds(tenor)
        zones = [  # (x0, x1, colour, opacity, label)
            (-lim, -enter, RV_COLOR[1], 0.35, "BUY (+1)"),
            (-enter, -exit_, RV_COLOR[1], 0.12, "keeps current call"),
            (-exit_, exit_, "#8c959f", 0.20, "NEUTRAL (0)"),
            (exit_, enter, RV_COLOR[-1], 0.12, "keeps current call"),
            (enter, lim, RV_COLOR[-1], 0.35, "SELL (-1)"),
        ]
        for x0, x1, color, op, label in zones:
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=1, fillcolor=color, opacity=op, line_width=0, layer="below", row=row, col=1)
            fig.add_annotation(x=(x0 + x1) / 2, y=0.86, text=label, showarrow=False, yanchor="middle", font=dict(size=11, color="#24292f"),
                               row=row, col=1)
        now, then = float(comp_all[tenor].iloc[-1]), float(comp_all[tenor].iloc[-1 - BREAKDOWN_LOOKBACK_DAYS])
        rv = result.signal[tenor]["rv"]
        fig.add_annotation(x=now, y=0.45, ax=then, ay=0.45, xref=f"x{row}" if row > 1 else "x", yref=f"y{row}" if row > 1 else "y",
                           axref=f"x{row}" if row > 1 else "x", ayref=f"y{row}" if row > 1 else "y",
                           showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="#57606a", standoff=10, text="")
        fig.add_trace(go.Scatter(x=[now], y=[0.45], mode="markers+text", text=[f"<b>{now:+.2f}</b>"], textposition="bottom center",
                                 marker=dict(size=20, color=RV_COLOR[rv], line=dict(color="white", width=2)),
                                 name="today (colour = current call)", showlegend=row == 1), row=row, col=1)
        fig.add_trace(go.Scatter(x=[then], y=[0.45], mode="markers", name="3 months ago", showlegend=row == 1,
                                 marker=dict(symbol="diamond", size=12, color="#24292f", line=dict(color="white", width=1.5))), row=row, col=1)
        label, dist = result.signal[tenor]["next_threshold"], result.signal[tenor]["distance"]
        fig.layout.annotations[row - 1].update(  # subplot titles are the first annotations
            text=f"<b>{tenor}: {RV_TEXT[rv]}</b>   composite {now:+.2f} (3 months ago {then:+.2f});  "
                                                    f"next change: {label}, needs {dist:+.2f}", x=0, xanchor="left", font=dict(size=13))
        fig.update_yaxes(range=[0, 1], showticklabels=False, showgrid=False, zeroline=False, row=row, col=1)
        fig.update_xaxes(range=[-lim, lim], dtick=0.25, tickformat="+.2f", showgrid=False, row=row, col=1)
    fig.update_xaxes(title_text="composite z  (negative = factors point to lower yields)", row=len(tenors), col=1)
    fig.update_layout(title=f"Where today's composite sits on the decision scale, as of {_asof(result)}", height=430,
                      template="plotly_white", margin=dict(t=120, b=70), legend=dict(orientation="h", x=1, xanchor="right", y=1.18))
    return fig


# ---------- section 8: appendix ----------

def fair_value_summary(result: RunResult) -> pd.DataFrame:
    fv = result.panel.fair_value
    fits = fv.fits
    last = fits.iloc[-1]
    rows = [{"stat": "last refit", "value": f"{fits.index[-1]:%Y-%m-%d}"},
            {"stat": "R-squared (trailing window)", "value": f"{last['r2']:.3f}" + ("  FLAG < 0.5" if fv.flagged else "")},
            {"stat": "observations", "value": f"{int(last['nobs'])}"},
            {"stat": "first fit", "value": f"{fits.index[0]:%Y-%m-%d}"}]
    coef = fits.drop(columns=["r2", "nobs"])
    for c in coef.columns:
        rows.append({"stat": f"coef {c} (last / 3y range)",
                     "value": f"{last[c]:+.3f}  [{coef[c].iloc[-36:].min():+.3f}, {coef[c].iloc[-36:].max():+.3f}]"})
    return pd.DataFrame(rows).set_index("stat")


def fig_fair_value(result: RunResult) -> go.Figure:
    fv = result.panel.fair_value
    start = settings()["backtest_start"]
    y = result.panel.yields["y10"].loc[start:]
    fitted = fv.fitted.loc[start:]
    fig = go.Figure([go.Scatter(x=y.index, y=y, name="10y actual"), go.Scatter(x=fitted.index, y=fitted, name="10y fitted (last fit)")])
    fig.add_trace(go.Scatter(x=fv.fits.index, y=fv.fits["r2"], name="R-squared (rhs)", yaxis="y2", line=dict(color="#8c959f", dash="dot")))
    fig.update_layout(title=f"10y fair value model, as of {_asof(result)}", yaxis_title="%",
                      yaxis2=dict(overlaying="y", side="right", range=[0, 1], tickformat=".1f", title="R-squared"), **LAYOUT)
    fig.update_layout(legend_x=1.08)
    return fig


def factor_library_table(result: RunResult) -> pd.DataFrame:
    lib = pd.DataFrame(load_factors()).set_index("id")
    meta = pd.DataFrame(result.panel.meta).T
    out = pd.DataFrame({
        "factor": lib["name"], "theme": lib["theme"], "sign": lib["sign"],
        "w_2y": lib["weight_2y"], "w_10y": lib["weight_10y"],
        "source used": meta["label"], "primary/fallback": meta["source"], "status": meta["status"],
        "last release": pd.to_datetime(meta["last_as_of"]).dt.strftime("%Y-%m-%d"),
        "max age (d)": meta["max_age_days"],
        "history from": pd.to_datetime(meta["first_valid"]).dt.strftime("%Y-%m-%d"),
        "true PIT share since 2005": meta["pit_share_bt"].astype(float).round(2),
    })
    return out


# ---------- section 6: sign stability ----------

def sign_stability(result: RunResult, tenor: str, start: str | None = None) -> pd.DataFrame:
    """Rolling 5y Spearman of each weighted factor's pressure vs the realized 6m change, at month-ends."""
    from ust_rv.scoring.scorecard import sign_check

    cfg = settings()
    start = start or cfg["backtest_start"]
    h, window = cfg["horizon_days"], cfg["sign_check"]["window_days"]
    pressure, dy = result.scores.pressure, result.scores.dy_h[tenor]
    ids = [f["id"] for f in load_factors() if f[f"weight_{tenor}"] > 0]
    idx = pressure.loc[start:].index.to_series()
    ends = idx.groupby(idx.index.to_period("M")).last()
    return pd.DataFrame({fid: [sign_check(pressure[fid].loc[:e], dy.loc[:e], h, window) for e in ends] for fid in ids}, index=pd.DatetimeIndex(ends.to_numpy()))


def fig_sign_stability(result: RunResult, rho: pd.DataFrame, tenor: str) -> go.Figure:
    names = pd.DataFrame(load_factors()).set_index("id")["name"]
    labels = [f"{i} {names[i]}" for i in rho.columns]
    fig = go.Figure(go.Heatmap(z=rho.T.to_numpy(), x=rho.index, y=labels, zmid=0, zmin=-0.6, zmax=0.6,
                               colorscale="RdBu", colorbar=dict(title="rho")))  # blue = behaves as signed
    warn = settings()["sign_check"]["warn_rho"]
    fig.update_layout(title=f"{tenor}: rolling 5y Spearman, pressure vs realized 6m change (WARN below {warn}), as of {_asof(result)}",
                      height=max(400, 22 * len(labels)), template="plotly_white", margin=dict(l=260))
    return fig


def warn_list(result: RunResult) -> pd.DataFrame:
    c = result.card
    w = c[(c["sign_2y"] == "WARN") | (c["sign_10y"] == "WARN")]
    return w[["factor", "theme", "rho_2y", "sign_2y", "rho_10y", "sign_10y"]].round(2)


# ---------- section 7: backtest summary ----------

STRATEGY_LABELS = {"always_long": "Always long (benchmark)", "momentum": "Momentum only (benchmark)"}


def backtest_metrics_frame(bt, model: str) -> pd.DataFrame:
    """Performance table for the live model and the two simple benchmarks."""
    def ci(x, fmt="{:.2f}"):
        return "" if not isinstance(x, tuple) or pd.isna(x[0]) else f"[{fmt.format(x[0])}, {fmt.format(x[1])}]"

    labels = {model: "Following the model", **STRATEGY_LABELS}
    m = bt.metrics.loc[[i for i in bt.metrics.index if i[1] in labels]].rename(index=labels, level=1)
    return pd.DataFrame({
        "IC": m["ic"].astype(float).round(2), "IC CI": m["ic_ci"].map(ci), "HAC t": m["hac_t_hac"].astype(float).round(1),
        "hit rate": m["hit_rate"].astype(float).round(2), "hit CI": m["hit_rate_ci"].map(ci),
        "Sharpe": m["overlay_sharpe"].astype(float).round(2), "Sharpe CI": m["sharpe_ci"].map(ci),
        "bp/yr": m["pnl_bp_per_year"].astype(float).round(0), "max adverse bp": m["max_adverse_bp"].astype(float).round(0),
        "neutral": m["neutral_share"].astype(float).round(2), "avg hold d": m["avg_hold_days"].astype(float).round(0),
        "turnover/yr": m["turnover_per_year"].astype(float).round(1),
    })


def factor_definitions(result: RunResult):
    """Full definition of every factor: series, publisher, transform and the reason for its sign."""
    lib = pd.DataFrame(load_factors()).set_index("id")
    meta = pd.DataFrame(result.panel.meta).T
    out = pd.DataFrame({
        "factor": lib["name"], "theme": lib["theme"],
        "sign": lib["sign"].map({1: "+", -1: "-"}),
        "used for": [", ".join(t for t in ("2y", "10y") if f[f"weight_{t}"] > 0) for _, f in lib.iterrows()],
        "source used today": meta["label"].fillna("unavailable"),
        "definition": lib["definition"],
    })
    return (
        out.style.set_properties(subset=["definition"], **{"text-align": "left", "white-space": "normal", "min-width": "520px"})
        .set_properties(subset=["factor", "source used today"], **{"text-align": "left"})
        .set_properties(subset=["sign", "used for", "theme"], **{"white-space": "nowrap", "text-align": "center"})
    )


# ---------- review visuals ----------

YIELD_COL = {"2y": "y2", "10y": "y10"}
WARN_COLOR = "#bf8700"


def _lib() -> pd.DataFrame:
    return pd.DataFrame(load_factors()).set_index("id")


BREAKDOWN_LOOKBACK_DAYS = 63  # 3 months of trading days


def _breakdown_figure(today: pd.Series, past: pd.Series, labels: list[str], outline: list[bool], past_date: pd.Timestamp,
                      title: str, note: str) -> go.Figure:
    """Horizontal bars of today's contributions (red = toward higher yields), outlined where flagged, with a diamond at the
    same item's contribution on `past_date`. Shared by the theme and factor breakdowns."""
    colors = [RV_COLOR[-1] if v > 0 else RV_COLOR[1] for v in today]
    fig = go.Figure(go.Bar(x=today, y=labels, orientation="h", name="today", marker=dict(
        color=colors, line=dict(color=[WARN_COLOR if o else "rgba(0,0,0,0)" for o in outline], width=3))))
    fig.add_trace(go.Scatter(x=past, y=labels, mode="markers", name=f"3 months ago ({past_date:%Y-%m-%d})",
                             marker=dict(symbol="diamond", size=10, color="#24292f", line=dict(color="white", width=1))))
    fig.update_layout(title=title, height=max(380, 22 * len(labels) + 200), template="plotly_white", margin=dict(l=320, b=120),
                      xaxis_title="contribution to composite z",
                      legend=dict(orientation="h", x=1, xanchor="right", y=1.0, yanchor="bottom"))
    # fixed pixel offset below the plot so the note clears the axis title at any chart height
    fig.add_annotation(text=note, xref="paper", yref="paper", x=0, y=0, yshift=-58, showarrow=False, align="left", yanchor="top")
    return fig


def _composite_raw(result: RunResult, tenor: str) -> pd.Series:
    from ust_rv.scoring.composite import weighted_average

    return weighted_average(result.scores.theme_z[tenor], result.scores.theme_weights[tenor])


def fig_theme_breakdown(result: RunResult, tenor: str) -> go.Figure:
    """Today's composite z split exactly into one term per theme plus the composite-mean constant."""
    from ust_rv.backtest.robustness import theme_terms_at

    s, zc = result.scores, settings()["zscore"]
    raw = _composite_raw(result, tenor)
    today, k = theme_terms_at(s.theme_z[tenor], s.theme_weights[tenor], raw, s.theme_z[tenor].index[-1], zc)
    past_date = s.theme_z[tenor].index[-1 - BREAKDOWN_LOOKBACK_DAYS]
    past, k_past = theme_terms_at(s.theme_z[tenor], s.theme_weights[tenor], raw, past_date, zc)
    today = today.sort_values()
    labels = [f"{t} (weight {s.theme_weights[tenor][t]:g})" for t in today.index]
    note = (f"Theme terms sum to {today.sum():+.2f}; composite 10y mean {k['composite_mean']:+.2f}; "
            f"composite z = {k['composite_z']:+.2f} (3 months ago: {k_past['composite_z']:+.2f}).")
    return _breakdown_figure(today, past.reindex(today.index), labels, [False] * len(today), past_date,
                             f"{tenor}: today's composite z by theme (red = toward higher yields), as of {_asof(result)}", note)


def fig_factor_breakdown(result: RunResult, tenor: str) -> go.Figure:
    """Today's composite z split exactly into one term per factor plus constant terms."""
    from ust_rv.backtest.robustness import composite_decomposition, factor_terms_at

    s, zc, lib = result.scores, settings()["zscore"], _lib()
    raw = _composite_raw(result, tenor)
    terms, k = composite_decomposition(result.card, s.theme_pressure[tenor], s.theme_weights[tenor], raw, tenor, zc)
    past_date = s.pressure.index[-1 - BREAKDOWN_LOOKBACK_DAYS]
    past, k_past = factor_terms_at(s.pressure, lib[f"weight_{tenor}"].astype(float), lib["theme"], s.theme_pressure[tenor],
                                   s.theme_weights[tenor], raw, past_date, zc)
    warn = (terms["sign_check"] == "WARN").tolist()
    labels = [f"{i} {r.factor}" + ("  [WARN]" if w else "") for (i, r), w in zip(terms.iterrows(), warn)]
    c = terms["contribution"]
    gross, warn_c = c.abs().sum(), c[terms["sign_check"] == "WARN"]
    note = (f"Factor terms sum to {c.sum():+.2f}; theme 10y means {k['theme_means']:+.2f}; "
            f"composite 10y mean {k['composite_mean']:+.2f}; composite z = {k['composite_z']:+.2f} "
            f"(3 months ago: {k_past['composite_z']:+.2f}).<br>"
            f"WARN factors (orange outline): net {warn_c.sum():+.2f}, {warn_c.abs().sum() / gross:.0%} of the gross factor contribution.")
    return _breakdown_figure(c, past.reindex(terms.index), labels, warn, past_date,
                             f"{tenor}: today's composite z by factor (red = toward higher yields), as of {_asof(result)}", note)


def fig_theme_heatmap(result: RunResult, tenor: str, years: int = 3) -> go.Figure:
    """Theme z by week (heatmap) with the composite z above it."""
    from plotly.subplots import make_subplots

    start = result.panel.date - pd.DateOffset(years=years)
    tz = result.scores.theme_z[tenor].loc[start:].resample("W-FRI").last()
    comp = result.scores.composite[tenor].loc[start:].resample("W-FRI").last()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.3, 0.7], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=comp.index, y=comp, name="composite z", line=dict(color=TENOR_COLOR[tenor])), row=1, col=1)
    fig.add_trace(go.Heatmap(z=tz.T.to_numpy(), x=tz.index, y=list(tz.columns), zmid=0, zmin=-3, zmax=3, colorscale="RdBu",
                             reversescale=True, colorbar=dict(title="theme z", y=0.35, len=0.7)), row=2, col=1)
    fig.update_layout(title=f"{tenor}: theme z by week (red = pushing yields higher) and the composite, last {years}y, as of {_asof(result)}",
                      height=520, template="plotly_white", showlegend=False)
    fig.update_yaxes(title_text="composite z", row=1, col=1)
    return fig


# Fixed-width buckets aligned with the decision thresholds (+/-0.25 exit, +/-0.50 entry): 0.25 wide inside +/-1,
# 0.5 wide out to +/-2, open-ended beyond.
OUTCOME_EDGES = (-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
THIN_BUCKET_DAYS = 120  # fewer days than this is less than one independent 6-month outcome


def outcome_buckets(composite: pd.Series, dy: pd.Series, start: str, edges=OUTCOME_EDGES) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    """Realized 6m yield change by composite bucket: percentiles per bucket, the bucket containing today's reading,
    and the underlying (composite, outcome, bucket) rows. Buckets with no days are dropped."""
    bins = [-np.inf, *edges, np.inf]
    names = [f"below<br>{edges[0]:+.2f}"] + [f"{a:+.2f}<br>to {b:+.2f}" for a, b in zip(edges[:-1], edges[1:])] + [f"above<br>{edges[-1]:+.2f}"]
    pairs = pd.concat([composite.rename("c"), dy.rename("dy")], axis=1, sort=True).loc[start:].dropna()
    pairs["bucket"] = pd.cut(pairs["c"], bins, labels=False)
    g = pairs.groupby("bucket")
    table = pd.DataFrame({
        "label": [names[b] for b in g.size().index], "n": g.size(),
        **{f"p{int(q * 100)}": g["dy"].quantile(q) for q in (0.10, 0.25, 0.50, 0.75, 0.90)},
        "share_up": g["dy"].apply(lambda s: (s > 0).mean()),
    })
    today_bucket = int(pd.cut([float(composite.dropna().iloc[-1])], bins, labels=False)[0])
    return table, today_bucket, pairs


def fig_conditional_outcomes(result: RunResult, tenor: str, start: str = "2009-01-01") -> go.Figure:
    """What the yield did over the next 6 months, grouped by how strong the composite reading was."""
    t, d, pairs = outcome_buckets(result.scores.composite[tenor], result.scores.dy_h[tenor], start)
    today = float(result.scores.composite[tenor].dropna().iloc[-1])
    labels = [f"{r.label}<br>n={int(r.n):,}" for r in t.itertuples()]
    fig = go.Figure()
    for (b, r), label in zip(t.iterrows(), labels):
        color = WARN_COLOR if b == d else TENOR_COLOR[tenor]
        thin = r.n < THIN_BUCKET_DAYS
        y = pairs.loc[pairs["bucket"] == b, "dy"]
        fig.add_trace(go.Box(y=y, x=[label] * len(y), name=label, boxpoints=False, marker_color=color, fillcolor=color,
                             line=dict(color=color, width=1.5), opacity=0.25 if thin else 0.6, showlegend=False))
    fig.add_trace(go.Scatter(x=labels, y=t["p50"], mode="lines+markers", name="median outcome (joined)",
                             line=dict(color="#cf222e", width=2.5), marker=dict(color="#cf222e", size=7)))
    fig.add_hline(y=0, line_color="#24292f", line_width=1)
    if d in t.index:
        r = t.loc[d]
        fig.add_annotation(x=labels[t.index.get_loc(d)], y=1.0, yref="paper", yanchor="bottom", showarrow=False,
                           font=dict(color="#9a6700", size=12), text=f"<b>today ({today:+.2f})</b>")
        summary = (f"<b>Readings like today's</b> ({r.label.replace('<br>', ' ')}, {int(r.n):,} days since {start[:4]}): over the next 6 months the {tenor} yield "
                   f"changed by a median of <b>{r.p50:+.0f} bp</b>;<br>the middle half of outcomes ran from {r.p25:+.0f} to {r.p75:+.0f} bp, "
                   f"and yields rose {r.share_up:.0%} of the time. Faded boxes have fewer than {THIN_BUCKET_DAYS} days: read them with caution.")
        fig.add_annotation(text=summary, xref="paper", yref="paper", x=0, y=0, yshift=-112, showarrow=False, align="left", yanchor="top")
    fig.update_layout(title=f"{tenor}: what the yield did next, by strength of the composite reading ({start[:4]}+), as of {_asof(result)}",
                      height=620, template="plotly_white", margin=dict(b=170, r=40),
                      xaxis=dict(categoryorder="array", categoryarray=labels, tickangle=0,
                                 title=dict(text="composite z bucket: factors point lower  <  >  factors point higher", standoff=12)),
                      yaxis_title="yield change over the next 6 months, bp",
                      legend=dict(orientation="h", x=0, y=1.08, yanchor="bottom"))
    return fig


def fig_entry_paths(result: RunResult, tenor: str, start: str = "2005-01-01", horizon: int = 126) -> go.Figure:
    """Yield path in the 6 months after every new buy or sell signal."""
    from plotly.subplots import make_subplots

    rv = result.scores.rv[tenor].loc[start:]
    y = result.panel.yields[YIELD_COL[tenor]].loc[start:]
    entries = rv[(rv != rv.shift()) & rv.isin([1, -1])]
    n_buy, n_sell = int((entries == 1).sum()), int((entries == -1).sum())
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=(
        f"after a BUY (+1), {n_buy} entries: want yields lower", f"after a SELL (-1), {n_sell} entries: want yields higher"))
    for col, side in ((1, 1), (2, -1)):
        paths = []
        for d in entries[entries == side].index:
            i = y.index.get_loc(d)
            path = (y.iloc[i:i + horizon + 1] - y.iloc[i]).to_numpy() * 100.0
            paths.append(path)
            fig.add_trace(go.Scatter(x=list(range(len(path))), y=path, mode="lines", line=dict(color=RV_COLOR[side], width=1), opacity=0.25,
                                     showlegend=False, hovertext=f"entry {d:%Y-%m-%d}", hoverinfo="text+y"), row=1, col=col)
        full = [p for p in paths if len(p) == horizon + 1]
        if full:
            med = pd.DataFrame(full).median()
            fig.add_trace(go.Scatter(x=med.index, y=med, mode="lines", line=dict(color="#24292f", width=3),
                                     name="median path (completed 6m windows)", showlegend=col == 1), row=1, col=col)
            end = pd.Series([p[-1] for p in full])
            right = float((end < 0).mean() if side == 1 else (end > 0).mean())
            # positioned in figure ("paper") coordinates from the panel's x-domain: axis-domain references are not
            # supported by every notebook renderer (VS Code drops them silently)
            x0 = fig.layout["xaxis" if col == 1 else f"xaxis{col}"].domain[0]
            fig.add_annotation(xref="paper", yref="paper", x=x0 + 0.01, y=0.99, xanchor="left", yanchor="top",
                               showarrow=False, align="left", bgcolor="white", bordercolor=RV_COLOR[side], borderwidth=2, borderpad=6,
                               text=f"<b>Right direction: {right:.0%} of the time</b><br>{len(full)} completed calls; median 6m move {end.median():+.0f} bp")
        fig.add_hline(y=0, line_color="#8c959f", line_width=1, row=1, col=col)
    fig.update_layout(title=f"{tenor}: yield change in the 6 months after each new signal (since {start[:4]}), as of {_asof(result)}",
                      height=460, template="plotly_white")
    fig.update_xaxes(title_text="trading days after entry")
    fig.update_yaxes(title_text="yield change since entry, bp", col=1)
    return fig


def fig_rolling_ic(result: RunResult, tenor: str, start: str = "2005-01-01") -> go.Figure:
    """Rolling 3y IC and IC by calendar year: when has the composite worked?"""
    from plotly.subplots import make_subplots

    from ust_rv.backtest import metrics

    comp, dy = result.scores.composite[tenor].loc[start:], result.scores.dy_h[tenor]
    roll = metrics.rolling_ic(comp, dy)
    pairs = pd.concat([comp.rename("c"), dy.rename("dy")], axis=1, sort=True).dropna()
    by_year = pairs.groupby(pairs.index.year).apply(lambda g: metrics.ic(g["c"], g["dy"]) if len(g) > 60 else float("nan"))
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12, subplot_titles=("rolling 3-year IC (month-ends)", "IC by calendar year"))
    fig.add_trace(go.Scatter(x=roll.index, y=roll, line=dict(color=TENOR_COLOR[tenor]), name="rolling 3y IC"), row=1, col=1)
    fig.add_trace(go.Bar(x=by_year.index.astype(str), y=by_year, marker_color=[RV_COLOR[1] if v > 0 else RV_COLOR[-1] for v in by_year.fillna(0)],
                         name="IC by year"), row=2, col=1)
    for r in (1, 2):
        fig.add_hline(y=0, line_color="#8c959f", line_width=1, row=r, col=1)
    fig.update_layout(title=f"{tenor}: when has the composite worked? (IC = rank correlation with the realized 6m change), as of {_asof(result)}",
                      height=560, template="plotly_white", showlegend=False)
    return fig


PNL_LINES = (  # (signal column, label, colour, width)
    ("model", "Following the model", None, 3.0),
    ("always_long", "Always long (benchmark)", "#8c959f", 1.6),
    ("momentum", "Momentum only (benchmark)", "#bf8700", 1.6),
)


def fig_pnl_attribution(result: RunResult, tenor: str, start: str = "2005-01-01") -> go.Figure:
    """Did following the model make money, and more than the simple alternatives? Cumulative P&L of three strategies,
    with the model's drawdown underneath."""
    from plotly.subplots import make_subplots

    from ust_rv.backtest import metrics

    bt = result.backtests[tenor]
    pnl = bt.pnl.loc[start:].rename(columns={result.signal[tenor]["model"]: "model"})
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.08,
                        subplot_titles=("Cumulative P&L (bp, per 1 unit of DV01)", "The model's drawdown: how far below its previous peak (bp)"))
    for col, label, color, width in PNL_LINES:
        cum = pnl[col].cumsum()
        color = color or TENOR_COLOR[tenor]
        fig.add_trace(go.Scatter(x=cum.index, y=cum, name=f"{label}: {cum.iloc[-1]:+,.0f} bp total", line=dict(color=color, width=width)), row=1, col=1)
    dd = metrics.drawdown(pnl["model"])
    fig.add_trace(go.Scatter(x=dd.index, y=dd, fill="tozeroy", line=dict(color=TENOR_COLOR[tenor], width=1), showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line_color="#8c959f", line_width=1, row=1, col=1)
    fig.update_layout(title=f"{tenor}: P&L from following the model vs two simple alternatives, since {start[:4]}, as of {_asof(result)}",
                      height=640, template="plotly_white", margin=dict(r=60, t=90, b=80),
                      legend=dict(orientation="h", x=0, y=-0.08, yanchor="top", font=dict(size=13)))
    return fig


def pnl_sources_table(result: RunResult, start: str = "2005-01-01") -> pd.DataFrame:
    """Average P&L per year of each strategy, split into direction (calling the move), carry and roll-down."""
    from ust_rv.backtest import engine, metrics

    rows = []
    for tenor, bt in result.backtests.items():
        _, scol, s_years, years, _ = engine.TENOR_SPEC[tenor]
        y = result.panel.yields.loc[start:]
        signals = {"Following the model": bt.signals[result.signal[tenor]["model"]], "Always long (benchmark)": bt.signals["always_long"],
                   "Momentum only (benchmark)": bt.signals["momentum"]}
        for label, rv in signals.items():
            comp = metrics.pnl_components(rv.loc[start:], y[YIELD_COL[tenor]], y[scol], s_years, years, y["ff"])
            n_years = (comp.index[-1] - comp.index[0]).days / 365.25
            per_year = comp.sum() / n_years
            rows.append({"tenor": tenor, "strategy": label, "calling the move": per_year["direction"], "carry": per_year["carry"],
                         "roll-down": per_year["roll"], "total": per_year.sum(), "Sharpe": metrics.sharpe(comp.sum(axis=1))})
    return pd.DataFrame(rows).set_index(["tenor", "strategy"]).round({"calling the move": 1, "carry": 1, "roll-down": 1, "total": 1, "Sharpe": 2})


def fig_credibility(result: RunResult, tenor: str, start: str = "2009-01-01", n_placebo: int = 300) -> go.Figure:
    """Left: IC of composites with random factor signs vs the real composite. Right: IC on independent (non-overlapping)
    6m windows for each possible start day."""
    from plotly.subplots import make_subplots

    from ust_rv.backtest import metrics
    from ust_rv.backtest.robustness import nonoverlapping_ic, placebo_ic

    s, lib, zc = result.scores, _lib(), settings()["zscore"]
    actual = metrics.ic(s.composite[tenor].loc[start:], s.dy_h[tenor])
    placebo = placebo_ic(s.z, lib[f"weight_{tenor}"].astype(float), lib["theme"], s.theme_weights[tenor], s.dy_h[tenor], zc, start, n=n_placebo)
    valid = placebo[np.isfinite(placebo)]
    beaten = int((valid < actual).sum())
    no = nonoverlapping_ic(s.composite[tenor], s.dy_h[tenor], settings()["horizon_days"], start)
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        f"random factor signs: real IC {actual:.2f} beats {beaten} of {len(valid)} draws",
        f"independent 6m windows (n = {int(no['n'].median())} each): median IC {no['ic'].median():.2f}"))
    fig.add_trace(go.Histogram(x=placebo, nbinsx=30, marker_color="#8c959f", name="random-sign composites"), row=1, col=1)
    fig.add_vline(x=actual, line_color=TENOR_COLOR[tenor], line_width=3, row=1, col=1)
    fig.add_trace(go.Histogram(x=no["ic"], nbinsx=25, marker_color=TENOR_COLOR[tenor], opacity=0.7, name="IC by start offset"), row=1, col=2)
    fig.add_vline(x=0, line_color="#24292f", line_width=1, row=1, col=2)
    fig.update_xaxes(title_text="IC", row=1, col=1)
    fig.update_xaxes(title_text="IC", row=1, col=2)
    fig.update_layout(title=f"{tenor}: how much to believe the IC ({start[:4]}+), as of {_asof(result)}", height=420,
                      template="plotly_white", showlegend=False)
    return fig


def fig_factor_ic(result: RunResult, tenor: str, start: str = "2005-01-01") -> go.Figure:
    """Each weighted factor's own record: rank correlation of its pressure with the realized 6m change."""
    from ust_rv.backtest import metrics

    card = result.card
    ids = card.index[(card[f"w_{tenor}"] > 0) & card["status"].isin(["ok", "stale"])]
    full = pd.Series({i: metrics.ic(result.scores.pressure[i].loc[start:], result.scores.dy_h[tenor]) for i in ids})
    recent = card.loc[ids, f"rho_{tenor}"].astype(float)
    order = full.sort_values().index
    labels = [f"{i} {card.loc[i, 'factor']} (w {card.loc[i, f'w_{tenor}']:.1f})" for i in order]
    fig = go.Figure([
        go.Bar(y=labels, x=full[order], orientation="h", name=f"since {start[:4]}",
               marker_color=[RV_COLOR[1] if v > 0 else RV_COLOR[-1] for v in full[order].fillna(0)]),
        go.Bar(y=labels, x=recent[order], orientation="h", name="trailing 5y", marker_color="rgba(36,41,47,0.35)"),
    ])
    fig.add_vline(x=0, line_color="#24292f", line_width=1)
    fig.add_vline(x=settings()["sign_check"]["warn_rho"], line_color=WARN_COLOR, line_dash="dot")
    fig.update_layout(title=f"{tenor}: each factor's record, pressure vs realized 6m change (green = behaves as signed), as of {_asof(result)}",
                      barmode="group", height=max(450, 26 * len(order) + 140), xaxis_title="rank correlation", margin=dict(l=330), **{
                          k: v for k, v in LAYOUT.items() if k != "margin"})
    return fig


def fig_factor_correlation(result: RunResult, tenor: str, start: str = "2009-01-01") -> go.Figure:
    """Correlation of factor pressures, clustered so near-duplicates sit together."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    card = result.card
    ids = card.index[(card[f"w_{tenor}"] > 0) & card["status"].isin(["ok", "stale"])]
    corr = result.scores.pressure[ids].loc[start:].corr()
    dist = squareform((1 - corr.abs()).clip(lower=0).to_numpy(), checks=False)
    order = corr.index[leaves_list(linkage(dist, "average"))]
    c = corr.loc[order, order]
    labels = [f"{i} {card.loc[i, 'factor']}" for i in order]
    fig = go.Figure(go.Heatmap(z=c.to_numpy(), x=list(order), y=labels, zmid=0, zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
                               colorbar=dict(title="corr")))
    fig.update_layout(title=f"{tenor}: correlation of factor pressures ({start[:4]}+), clustered: dark blocks are factors saying the same thing, as of {_asof(result)}",
                      height=max(500, 22 * len(order) + 160), template="plotly_white", margin=dict(l=300), yaxis=dict(autorange="reversed"))
    return fig


def fig_provenance(result: RunResult) -> go.Figure:
    """Per factor: approximated history, true point-in-time history, and when it starts counting in the composite."""
    panel, z = result.panel, result.scores.z
    lib = _lib()
    end = panel.date
    rows = []
    for fid in reversed(lib.index):
        v = panel.values[fid].dropna()
        label = f"{fid} {lib.loc[fid, 'name']}"
        if v.empty:
            rows.append((label, None, None, None, panel.meta[fid]["status"]))
            continue
        pit = panel.pit[fid].reindex(v.index).astype("boolean").fillna(False).astype(bool)
        approx_end = pit[pit].index[0] if pit.any() else end
        zs = z[fid].dropna()
        rows.append((label, v.index[0], approx_end, zs.index[0] if not zs.empty else None, panel.meta[fid]["status"]))
    fig = go.Figure()
    ms = lambda a, b: (b - a).total_seconds() * 1000  # noqa: E731
    for label, first, approx_end, z_start, status in rows:
        if first is None:
            fig.add_trace(go.Bar(y=[label], x=[0], base=[end], orientation="h", marker_color="#d0d7de", showlegend=False,
                                 hovertext=status))
            continue
        if approx_end > first:
            fig.add_trace(go.Bar(y=[label], x=[ms(first, approx_end)], base=[first], orientation="h", marker_color="#d4a72c",
                                 name="approximated (not point-in-time)", legendgroup="a", showlegend=False))
        if end > approx_end:
            fig.add_trace(go.Bar(y=[label], x=[ms(approx_end, end)], base=[approx_end], orientation="h", marker_color="#2da44e",
                                 name="true point-in-time", legendgroup="p", showlegend=False))
        if z_start is not None:
            fig.add_trace(go.Scatter(y=[label], x=[z_start], mode="markers", marker=dict(symbol="line-ns", size=14, color="#24292f",
                                     line=dict(width=3, color="#24292f")), name="enters composite (5y of history)", legendgroup="z", showlegend=False))
    for name, color in (("approximated (not point-in-time)", "#d4a72c"), ("true point-in-time", "#2da44e"), ("unavailable", "#d0d7de")):
        fig.add_trace(go.Bar(y=[None], x=[None], marker_color=color, name=name))
    fig.add_trace(go.Scatter(y=[None], x=[None], mode="markers", marker=dict(symbol="line-ns", size=14, line=dict(width=3, color="#24292f")),
                             name="enters composite (5y of history)"))
    fig.add_vline(x=pd.Timestamp(settings()["backtest_start"]), line_dash="dot", line_color="#57606a")
    fig.update_layout(title=f"Data provenance by factor: approximated vs true point-in-time history, as of {_asof(result)}", barmode="overlay",
                      height=24 * len(rows) + 200, template="plotly_white", xaxis=dict(type="date"), margin=dict(l=300, b=90),
                      legend=dict(orientation="h", y=-0.04, yanchor="top", x=0))
    return fig


def forward_ic_table(result: RunResult) -> pd.DataFrame:
    """IC of the live composite against the raw 6m yield change and against the change versus the 6m-forward yield."""
    from ust_rv.backtest import engine, metrics
    from ust_rv.backtest.forward_target import forward_target

    h = settings()["horizon_days"]
    rows = []
    for tenor in TENORS:
        fwd = forward_target(result.panel.yields, tenor, h)
        comp, dy = result.scores.composite[tenor], result.scores.dy_h[tenor]
        w = engine.windows(engine.pit_share(result.panel, tenor))
        wins = {"2005+": w["2005+"], "2012+": w["2005+"][w["2005+"] >= "2012-01-01"], "PIT>=90%": w["PIT>=90%"]}
        for name, idx in wins.items():
            c = comp.reindex(idx)
            frame = pd.concat([c.rename("c"), fwd.rename("d")], axis=1).dropna()
            lo, hi = metrics.block_bootstrap(frame, lambda f: metrics.ic(f["c"], f["d"]), block=126, n_draws=300)
            rows.append({"tenor": tenor, "window": name, "IC vs raw 6m change": round(metrics.ic(c, dy), 2),
                         "IC vs move beyond forwards": round(metrics.ic(c, fwd), 2), "95% CI (beyond forwards)": f"[{lo:.2f}, {hi:.2f}]"})
    return pd.DataFrame(rows).set_index(["tenor", "window"])
