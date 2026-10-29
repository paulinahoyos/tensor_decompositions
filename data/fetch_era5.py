"""ERA5 anomalies from WeatherBench 2 (paper Section 6.3).

Data: ERA5 on the 64 x 32 equiangular grid (5.625 degrees, no poles), 6-hourly,
from the public WeatherBench 2 bucket (anonymous access):
    gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr
Climatology (1990-2019, 6-hourly by day of year):
    gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr

Selection: the year 2019, one value per day at 00 UTC (365 days); variables
geopotential, temperature, u and v wind, specific humidity; all 13 pressure levels.
Anomalies = value - climatology(day of year, 00 UTC), then each (variable, level)
pair is divided by the standard deviation of its anomalies over (lon, lat, time).

Output: data/processed/era5_2019.npz with
    X      (lon 64, lat 32, level 13, var 5, time 365) float32, standardized anomalies
    lon, lat, level, variables, dates, std (var, level)
About 200 MB of the about 1.3 GB transferred from the bucket is kept (the chunking
forces whole chunks to be read: 100 time steps per ERA5 chunk, one chunk per
climatology variable).

Usage:
    pip install -e ".[data]"
    python data/fetch_era5.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ERA5 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
CLIM = ("gs://weatherbench2/datasets/era5-hourly-climatology/"
        "1990-2019_6h_64x32_equiangular_conservative.zarr")
VARIABLES = ["geopotential", "temperature", "u_component_of_wind", "v_component_of_wind",
             "specific_humidity"]
HERE = Path(__file__).resolve().parent


def main():
    import xarray as xr

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", default="2019")
    ap.add_argument("--hour", type=int, default=0, help="UTC hour sampled each day")
    ap.add_argument("--out", default=str(HERE / "processed" / "era5_2019.npz"))
    args = ap.parse_args()

    opts = dict(storage_options={"token": "anon"}, chunks=None)
    ds = xr.open_zarr(ERA5, **opts)
    clim = xr.open_zarr(CLIM, **opts)
    print("ERA5 dims:", dict(ds.sizes))
    print("climatology dims:", dict(clim.sizes))

    t = ds.time.sel(time=slice(f"{args.year}-01-01", f"{args.year}-12-31"))
    t = t[t.dt.hour == args.hour]
    print(f"{t.size} days in {args.year} at {args.hour:02d} UTC")
    doy = t.dt.dayofyear

    fields = []
    for v in VARIABLES:
        a = ds[v].sel(time=t).transpose("longitude", "latitude", "level", "time").load()
        c = clim[v].sel(hour=args.hour, dayofyear=doy).transpose("longitude", "latitude", "level", "time").load()
        anom = a.values.astype(np.float64) - c.values.astype(np.float64)
        print(f"  {v}: {a.shape}, mean anomaly {anom.mean():.3g}")
        fields.append(anom)
    X = np.stack(fields, axis=3)                     # (lon, lat, level, var, time)
    std = X.std(axis=(0, 1, 4))                      # (level, var)
    X = X / std[None, None, :, :, None]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X.astype(np.float32), lon=ds.longitude.values, lat=ds.latitude.values,
             level=ds.level.values, variables=np.array(VARIABLES), dates=t.values.astype("datetime64[D]"),
             std=std.T)
    print(f"saved {out}: X {X.shape} float32, {X.astype(np.float32).nbytes / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
