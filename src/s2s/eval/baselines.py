"""Reference forecasts (task #4). Every model result must beat these.

  climatology : predict the seasonal cycle => zero anomaly. The HARD baseline.
  persistence : carry the most recent observed anomaly forward to all leads.

Beating persistence but not climatology is the classic false win — report both.
"""
from __future__ import annotations
import xarray as xr


def _lead_dim(cfg) -> xr.DataArray:
    return xr.DataArray(list(cfg.data.lead_weeks), dims="lead", name="lead")


def climatology_forecast(init_time, clim, cfg) -> xr.Dataset:
    """Zero-anomaly forecast for weeks 1-6 (i.e. predict the climatology).

    `clim` only supplies the variable set/template (its `dayofyear` dim is
    dropped) -- in anomaly space, "predict the climatology" IS "predict zero",
    by construction of `to_anomaly`.
    """
    template = clim.isel(dayofyear=0, drop=True)
    zero = xr.zeros_like(template)
    forecast = xr.concat([zero for _ in cfg.data.lead_weeks], dim=_lead_dim(cfg))
    return forecast.expand_dims(time=[init_time])


def persistence_forecast(init_anomaly, cfg) -> xr.Dataset:
    """Hold the latest observed anomaly constant across all lead weeks."""
    forecast = xr.concat([init_anomaly for _ in cfg.data.lead_weeks], dim=_lead_dim(cfg))
    return forecast


def _week_of_year(times):
    """ISO week index in [1, 53] from a datetime64 array."""
    import numpy as np
    return xr.DataArray(times).dt.isocalendar().week.values.astype(np.int64)


def climatology_woy_ensemble(train_weekly, target_woy: int, window: int = 3,
                             member_dim: str = "member"):
    """Seasonality-aware probabilistic climatology for ONE target week-of-year.

    The honest Phase-B reference (ADR-0001 / phase-b-plan): pools TRAIN weekly
    anomalies whose week-of-year lies within +/-`window` weeks (circular over the
    ~52-week year) of the target's week-of-year, returned along `member`.

    Why windowed, not all-pooled: even in de-seasonalized anomaly space the
    *variance* of weekly anomalies is itself seasonal (monsoon precip spread >>
    dry-season). A +/-3-week window keeps the reference local in season -- an
    honest, harder bar -- while retaining ~7 weeks x #train-years members for a
    stable CRPS. Train-only => no leakage.

    train_weekly : (time, lat, lon) DataArray of TRAIN weekly anomalies (one var).
    target_woy   : int week-of-year (1..53) of the verification target.
    """
    import numpy as np
    woy = _week_of_year(train_weekly["time"].values)
    d = np.abs(woy - int(target_woy))
    d = np.minimum(d, 52 - d)            # circular distance on a 52-week ring
    sel = np.where(d <= window)[0]
    pool = train_weekly.isel(time=sel)
    return pool.rename({"time": member_dim})


def _decimal_year(times):
    """datetime64/Timestamp (scalar or array) -> decimal year, e.g. 2020.37.

    Same origin/units are used for both the trend fit and the per-member shift,
    so only the SLOPE matters — any consistent linear time axis would give the
    identical detrended reference.
    """
    import numpy as np
    import pandas as pd
    idx = pd.DatetimeIndex(np.atleast_1d(np.asarray(times, dtype="datetime64[ns]")))
    return (idx.year + (idx.dayofyear - 1) / 365.25).values.astype(np.float64)


def fit_linear_trend(train_weekly, dim: str = "time"):
    """Per-gridpoint OLS slope of TRAIN weekly anomalies vs time (decimal years).

    train_weekly : (time, lat, lon) TRAIN-only anomalies for ONE variable.
    Returns a (lat, lon) DataArray of slope (anomaly units per year). Train-only
    by construction — the caller passes the train split, so no val/test leakage.
    """
    x = _decimal_year(train_weekly[dim].values)          # (T,)
    xm = x - x.mean()
    denom = float((xm ** 2).sum())
    y = train_weekly.transpose(dim, "latitude", "longitude").values  # (T, lat, lon)
    ym = y - y.mean(axis=0, keepdims=True)
    slope = (xm[:, None, None] * ym).sum(axis=0) / denom  # (lat, lon)
    return xr.DataArray(
        slope,
        dims=("latitude", "longitude"),
        coords={"latitude": train_weekly["latitude"], "longitude": train_weekly["longitude"]},
        name="trend_slope",
    )


def climatology_woy_trend_ensemble(
    train_weekly, target_woy: int, target_time, window: int = 3,
    member_dim: str = "member", trend=None,
):
    """Trend-aware probabilistic climatology for ONE target week & date (Fix 3 / C1).

    Identical to `climatology_woy_ensemble` (same +/-`window` WOY pool of TRAIN
    weekly anomalies) but each pooled member is shifted by

        (trend(target_time) - trend(member_source_date))
      =  slope * (decimal_year(target_time) - decimal_year(member_date))

    per gridpoint, using a per-gridpoint linear trend fitted on TRAIN years only.
    This recentres every member from its own epoch onto the target date's epoch,
    removing the warming-trend advantage that the raw pool (centred on the
    1979-2012 mean) hands a model over a 2018-2023 test period. If the model's
    "skill" is really just the trend, CRPSS against THIS reference collapses.

    train_weekly : (time, lat, lon) TRAIN-only anomalies for one variable.
    target_time  : the verification target date (scalar datetime64/Timestamp).
    trend        : optional precomputed (lat, lon) slope from fit_linear_trend;
                   fitted here on `train_weekly` if omitted (pass it in to avoid
                   refitting inside a per-sample loop).
    """
    pool = climatology_woy_ensemble(train_weekly, target_woy, window=window, member_dim=member_dim)
    if trend is None:
        trend = fit_linear_trend(train_weekly)
    t_target = float(_decimal_year(target_time)[0])
    t_member = _decimal_year(pool[member_dim].values)          # (member,)
    # shift[m, lat, lon] = slope[lat, lon] * (t_target - t_member[m])
    dt = xr.DataArray(t_target - t_member, dims=(member_dim,), coords={member_dim: pool[member_dim]})
    shift = trend * dt                                          # broadcast -> (member, lat, lon)
    return pool + shift


class PolyTrend:
    """Per-gridpoint polynomial trend fitted on TRAIN years only (Fix 3 sensitivity).

    Generalises fit_linear_trend to degree>=1 so we can test whether the residual
    lead-independent CRPSS 'floor' against the linear-detrended reference is really
    linear-trend-residual (nonlinear warming) or a genuine calibrated-spread effect.
    Coefficients are in increasing powers of the CENTRED decimal year (t - x_mean),
    fitted by least squares. Train-only by construction.
    """

    def __init__(self, coeffs, x_mean, latitude, longitude):
        self.coeffs = coeffs                 # (degree+1, lat, lon), increasing powers
        self.x_mean = float(x_mean)
        self.latitude = latitude
        self.longitude = longitude

    def evaluate(self, times):
        import numpy as np
        xc = _decimal_year(times) - self.x_mean            # (n,)
        powers = np.stack([xc ** k for k in range(self.coeffs.shape[0])], axis=0)  # (deg+1, n)
        # (deg+1, n) x (deg+1, lat, lon) -> (n, lat, lon)
        return np.einsum("kn,klm->nlm", powers, self.coeffs)


def fit_poly_trend(train_weekly, degree: int = 2, dim: str = "time") -> "PolyTrend":
    """Least-squares per-gridpoint polynomial trend of TRAIN weekly anomalies vs
    centred decimal year. degree=1 matches fit_linear_trend."""
    import numpy as np
    x = _decimal_year(train_weekly[dim].values)
    x_mean = x.mean()
    xc = x - x_mean
    V = np.vander(xc, degree + 1, increasing=True)         # (T, degree+1)
    y = train_weekly.transpose(dim, "latitude", "longitude").values  # (T, lat, lon)
    T, H, W = y.shape
    coeffs, *_ = np.linalg.lstsq(V, y.reshape(T, H * W), rcond=None)  # (degree+1, H*W)
    return PolyTrend(coeffs.reshape(degree + 1, H, W), x_mean,
                     train_weekly["latitude"], train_weekly["longitude"])


def climatology_woy_polytrend_ensemble(
    train_weekly, target_woy: int, target_time, window: int = 3,
    member_dim: str = "member", poly: "PolyTrend | None" = None, degree: int = 2,
):
    """Like climatology_woy_trend_ensemble but with a degree>=1 polynomial trend.

    Each pooled member is shifted by poly(target_time) - poly(member_source_date).
    degree=1 reproduces the linear trend-aware reference (asserted in tests).
    """
    import numpy as np
    pool = climatology_woy_ensemble(train_weekly, target_woy, window=window, member_dim=member_dim)
    if poly is None:
        poly = fit_poly_trend(train_weekly, degree=degree)
    val_target = poly.evaluate(np.atleast_1d(np.asarray(target_time, dtype="datetime64[ns]")))[0]  # (lat,lon)
    val_member = poly.evaluate(pool[member_dim].values)                                             # (M,lat,lon)
    shift = xr.DataArray(
        val_target[None, :, :] - val_member,
        dims=(member_dim, "latitude", "longitude"),
        coords={member_dim: pool[member_dim], "latitude": pool["latitude"], "longitude": pool["longitude"]},
    )
    return pool + shift
