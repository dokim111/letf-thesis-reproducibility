"""Predefined additions for the initial undergraduate graduation-thesis draft.

Run from any directory: python3 stats_graduation_extension.py
No new market data, no random numbers, and no refitting after evaluation exclusions.
All fits use only targets completed by the forecast origin (s+h <= t).
The existing manuscripts' forecasts are reproduced as a numerical gate.
"""
from pathlib import Path
import hashlib
import json
import os

# Small repeated least-squares fits are faster and deterministic with one BLAS thread.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "graduation_extension_results"
OUT.mkdir(exist_ok=True)
START, LISTING, EVENT, MIN_TRAIN, EPS = 20150615, 20260527, 20260731, 750, 1e-8
FLOORS = (1e-8, 1e-6)
DATA = ROOT / "krx_stock_daily_long_20000104_20261113.csv"
REFERENCE = ROOT / "stats_addendum_results.csv"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def roll(x, w):
    return pd.Series(x).rolling(w).mean().to_numpy()


def g(beta, r):
    return beta * np.log1p(r) - np.log1p(beta * r)


def nw_t(z, lag):
    z = np.asarray(z)
    u = z - z.mean()
    n = len(z)
    variance = u @ u / n
    for k in range(1, min(lag, n - 1) + 1):
        variance += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    return float(z.mean() / np.sqrt(variance / n)) if variance > 0 else np.nan


def mse(y, pred):
    return float(np.mean((y - pred) ** 2))


def qlike(y, forecast):
    assert np.all(y > 0) and np.all(forecast > 0)
    ratio = y / forecast
    return ratio - np.log(ratio) - 1


design = {
    "status": "post-hoc extension; no untouched holdout is claimed",
    "data_sha256": sha(DATA),
    "reference_sha256": sha(REFERENCE),
    "start_date": START,
    "listing_date": LISTING,
    "event_date": EVENT,
    "horizons": [5, 20],
    "betas": [2, -2],
    "minimum_training_setting": MIN_TRAIN,
    "first_origin_index": "22 + MIN_TRAIN + h (zero based, same as original)",
    "training_rows": "21 <= s <= t-h",
    "pre_strict": "target end date < listing date",
    "post": "forecast origin date >= listing date",
    "event_exclusion": "event in future target or preceding 22-day input; scores only, not refitting",
    "epsilon": EPS,
    "main_asymmetry": "mean r^3/(mean r^2+EPS)^(3/2), windows 5,22",
    "downside_information": "mean(r^2*1[r<0])/(mean(r^2)+EPS), windows 1,5,22",
    "main_models": ["M1", "M2", "MD", "MDK"],
    "log_to_level": "exp(log forecast)*mean(exp(in-sample log residuals))-EPS; mature training rows only",
    "positive_projection_floors": list(FLOORS),
    "comparison_losses": ["MSE on positive level forecasts", "log-MSE on same level forecasts", "QLIKE"],
    "inference": "CW only for nested original log-target squared errors; no CW tests for QLIKE or retransformed forecasts",
    "no_new_model_selection": True,
}
(OUT / "design.json").write_text(json.dumps(design, ensure_ascii=False, indent=2) + "\n")

df = pd.read_csv(DATA, encoding="utf-8-sig", dtype={"code": str})
df["r"] = df.close / (df.close - df.change_from_base_price) - 1
reference = pd.read_csv(REFERENCE)
semi_rows, loss_rows, gain_rows, prediction_rows, validation_rows = [], [], [], [], []

for stock, sub in df.groupby("stock"):
    sub = sub[sub.date >= START].reset_index(drop=True)
    dates, r = sub.date.to_numpy(), sub.r.to_numpy()
    n = len(r)
    rv = {w: roll(r ** 2, w) for w in (1, 5, 22)}
    down = {w: roll(r ** 2 * (r < 0), w) for w in (1, 5, 22)}
    m22 = roll(r, 22)
    variance_log = np.column_stack([np.ones(n)] + [np.log(rv[w] + EPS) for w in (1, 5, 22)] + [m22])
    variance_raw = np.column_stack([np.ones(n)] + [rv[w] for w in (1, 5, 22)] + [m22])
    cubic_normalized = np.column_stack([roll(r ** 3, w) / (rv[w] + EPS) ** 1.5 for w in (5, 22)])
    cubic_raw = np.column_stack([roll(r ** 3, w) for w in (1, 5, 22)])
    downside_share = np.column_stack([down[w] / (rv[w] + EPS) for w in (1, 5, 22)])
    designs = {
        "log_M1": variance_log,
        "log_M2": np.column_stack([variance_log, cubic_normalized]),
        "log_MD": np.column_stack([variance_log, downside_share]),
        "log_MDK": np.column_stack([variance_log, downside_share, cubic_normalized]),
        "raw_M1": variance_raw,
        "raw_M2": np.column_stack([variance_raw, cubic_raw]),
    }
    for beta in (2, -2):
        assert np.all(1 + beta * r > 0)
    daily_g = np.column_stack([g(beta, r) for beta in (2, -2)])
    sums_g = np.vstack([np.zeros(2), np.cumsum(daily_g, axis=0)])
    base_daily = np.column_stack([roll(daily_g[:, j], 22) for j in (0, 1)])
    event_i = int(np.flatnonzero(dates == EVENT)[0])
    for h in (5, 20):
        target = np.full((n, 2), np.nan)
        target[: n - h] = sums_g[1 + h : n + 1] - sums_g[1 : n - h + 1]
        assert np.nanmin(target) > 0
        log_target = np.log(target + EPS)
        origins = np.arange(22 + MIN_TRAIN + h, n - h)
        base = h * base_daily[origins]
        base_log = np.log(base + EPS)
        end_dates = dates[origins + h]
        affected = (origins + h >= event_i) & (origins <= event_i + 21)
        masks = {
            "pre_strict": end_dates < LISTING,
            "post": dates[origins] >= LISTING,
            "post_no_event": (dates[origins] >= LISTING) & ~affected,
        }
        fitted = {}
        # Both leverage targets share X; solve their coefficients in a single call.
        Y_all = np.column_stack([target, log_target])
        for label, X in designs.items():
            pred = np.empty((len(origins), 4))
            smear = np.empty((len(origins), 2))
            for k, t in enumerate(origins):
                training = slice(21, int(t - h + 1))
                coef = np.linalg.lstsq(X[training], Y_all[training], rcond=None)[0]
                pred[k] = X[t] @ coef
                residual_log = log_target[training] - X[training] @ coef[:, 2:]
                smear[k] = np.exp(residual_log).mean(axis=0)
            assert np.isfinite(pred).all() and np.isfinite(smear).all()
            fitted[label] = {"level": pred[:, :2], "log": pred[:, 2:], "smear": smear}
        # Exact reproduction of the existing four original target/design combinations.
        for vk in ("log", "raw"):
            ak = "norm2" if vk == "log" else "raw3"
            for transform in ("level", "log"):
                for j, beta in enumerate((2, -2)):
                    target_name = ("G" if transform == "level" else "logG") + "(%+d)" % beta
                    y = target[origins, j] if transform == "level" else log_target[origins, j]
                    p1 = fitted[vk + "_M1"][transform][:, j]
                    p2 = fitted[vk + "_M2"][transform][:, j]
                    for sample in ("pre_strict", "post"):
                        m = masks[sample]
                        gain = 1 - mse(y[m], p2[m]) / mse(y[m], p1[m])
                        query = reference[(reference.stock == stock) & (reference.h == h) & (reference.target == target_name) & (reference.variance_block == vk) & (reference.asym_block == ak) & (reference["sample"] == sample)]
                        assert len(query) == 1
                        error = abs(gain - float(query.iloc[0].gain_M2_over_M1))
                        assert error < 1e-8, (stock, h, beta, vk, transform, sample, error)
                        validation_rows.append(dict(stock=stock, h=h, beta=beta, variance=vk, transform=transform, sample=sample, error=error))
        # The main semivariance extension is scored in the original log-target units.
        for j, beta in enumerate((2, -2)):
            y = log_target[origins, j]
            for sample, m in masks.items():
                for narrow, broad in (("M1", "M2"), ("M1", "MD"), ("MD", "MDK")):
                    pn = fitted["log_" + narrow]["log"][:, j]
                    pb = fitted["log_" + broad]["log"][:, j]
                    en, eb = y[m] - pn[m], y[m] - pb[m]
                    cw = en ** 2 - eb ** 2 + (pn[m] - pb[m]) ** 2
                    semi_rows.append(dict(stock=stock, h=h, beta=beta, sample=sample, n=int(m.sum()), narrower=narrow, broader=broad, mse_narrow=float(np.mean(en ** 2)), mse_broad=float(np.mean(eb ** 2)), improvement=1 - np.mean(eb ** 2) / np.mean(en ** 2), cw_h1=nw_t(cw, h-1) if sample != "post_no_event" else np.nan, cw_40=nw_t(cw, 40) if sample != "post_no_event" else np.nan))
        # Common positive G scale. Log forecasts get training-residual smearing.
        # Log-MSE here is evaluated on these same positive mean-oriented forecasts;
        # it is distinct from the original log-target OLS comparison above.
        positive_models = {"M0": base}
        for label, fit in fitted.items():
            positive_models[label + "_fit_level"] = fit["level"]
            positive_models[label + "_fit_log_smear"] = np.exp(fit["log"]) * fit["smear"] - EPS
        for j, beta in enumerate((2, -2)):
            y = target[origins, j]
            for model_name, original_prediction in positive_models.items():
                p_unprojected = original_prediction[:, j]
                for floor in FLOORS:
                    pred = np.maximum(p_unprojected, floor)
                    for sample, m in masks.items():
                        ls = {"mse": (y[m] - pred[m]) ** 2,
                              "log_mse": (np.log(y[m] + EPS) - np.log(pred[m] + EPS)) ** 2,
                              "qlike": qlike(y[m], pred[m])}
                        loss_rows.append(dict(stock=stock, h=h, beta=beta, sample=sample, n=int(m.sum()), model=model_name, floor=floor, mse=float(ls["mse"].mean()), log_mse=float(ls["log_mse"].mean()), qlike=float(ls["qlike"].mean()), n_nonpositive=int((p_unprojected[m] <= 0).sum()), n_floor=int((p_unprojected[m] < floor).sum()), min_raw_pred=float(p_unprojected[m].min()), mean_raw_pred=float(p_unprojected[m].mean())))
                # Only primary floor need be stored per forecast; other scores reproduce it.
                pp = np.maximum(p_unprojected, FLOORS[0])
                for k, t in enumerate(origins):
                    prediction_rows.append(dict(stock=stock, h=h, beta=beta, model=model_name, date=int(dates[t]), end_date=int(dates[t+h]), y=float(y[k]), raw_prediction=float(p_unprojected[k]), positive_prediction=float(pp[k]), event_affected=bool(affected[k])))
        print(stock, h, "completed", flush=True)

semis = pd.DataFrame(semi_rows)
losses = pd.DataFrame(loss_rows)
for keys, q in losses.groupby(["stock", "h", "beta", "sample", "floor"]):
    bymodel = q.set_index("model")
    pairs = [("log_M1_fit_log_smear", "log_M2_fit_log_smear"),
             ("log_M1_fit_log_smear", "log_MD_fit_log_smear"),
             ("log_MD_fit_log_smear", "log_MDK_fit_log_smear"),
             ("raw_M1_fit_level", "raw_M2_fit_level"),
             ("log_M1_fit_level", "log_M2_fit_level"),
             ("raw_M1_fit_log_smear", "raw_M2_fit_log_smear")]
    for smaller, larger in pairs:
        a, b = bymodel.loc[smaller], bymodel.loc[larger]
        row = dict(zip(["stock", "h", "beta", "sample", "floor"], keys))
        row.update(n=int(a.n), smaller=smaller, larger=larger)
        for loss in ("mse", "log_mse", "qlike"):
            row[loss + "_improvement"] = 1 - float(b[loss])/float(a[loss])
        gain_rows.append(row)
semis.to_csv(OUT / "semivariance_log_results.csv", index=False, encoding="utf-8-sig")
losses.to_csv(OUT / "common_scale_losses.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(gain_rows).to_csv(OUT / "common_scale_improvements.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(prediction_rows).to_csv(OUT / "forecasts.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}, encoding="utf-8-sig")
pd.DataFrame(validation_rows).to_csv(OUT / "original_forecast_reproduction.csv", index=False)
validation = {"number_of_original_comparisons": len(validation_rows), "max_abs_original_gain_difference": max(row["error"] for row in validation_rows), "input_unchanged": sha(DATA) == design["data_sha256"] and sha(REFERENCE) == design["reference_sha256"], "design": design, "all_new_forecasts_finite": True, "event_exclusions_refit": False}
assert validation["input_unchanged"]
(OUT / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
print("Validation:", json.dumps(validation, ensure_ascii=False), flush=True)
print("SEMIVARIANCE LOG RESULTS\n", semis.round(5).to_string(index=False), flush=True)

