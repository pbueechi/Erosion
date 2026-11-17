"""
Side-by-side Sentinel-2 RGB of one sampled field at two timesteps.

Uses the same inputs as the pipelines (main.py / compute_cfactor_pixels.py CONFIG):
  - lnf_dir       : LNF field polygons lnf{yr}.gpkg (column ``id``, ``lnf_code``)
  - samples_path  : sampled fields (poly_id, yr, lnf_code, polygon_geom), EPSG:32632
  - s2_grid_path  : S2 tile grid (columns left, top)
  - s2_dir        : S2 zarr tiles named S2_{left}_{top}_{year}*.zarr

Two poly_id definitions exist in the project, selected with ``id_source``:
  - 'lnf' (default): poly_id == LNF ``id`` column. Used by the per-pixel
    inference / comparison products (compute_cfactor_pixels*.py,
    extract_fc_pixels.py, cfactor_analysis/compare_products.py).
  - 'samples': poly_id == row index of the yearly LNF file, as written by
    sample_FC.py into samplesv2.pkl and carried into the calibration/training
    outputs. Only unique *within a year*.

The agricultural year (Jul yr-1 .. Jun yr) is inferred from ``timestep1``
unless ``yr`` is given.

Usage:
    python plot_s2_rgb.py 12557 2021-04-15 2021-06-20
    python plot_s2_rgb.py 12345 2021-04-15 2021-06-20 --id-source samples
or from Python:
    from plot_s2_rgb import plot_s2_rgb_pair
    plot_s2_rgb_pair(12557, '2021-04-15', '2021-06-20')
"""
from __future__ import annotations

import argparse
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Same paths as main.py CONFIG (copied rather than imported: importing main
# pulls in sample_FC, which loads torch and the FC models).
SAMPLES_PATH = 'samplesv2.pkl'
LNF_DIR      = '~/mnt/eo-nas1/data/landuse/raw'
S2_GRID_PATH = '~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles_CH.shp'
S2_DIR       = '~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH'
S2_CRS       = 'EPSG:32632'

RGB_BANDS = ['s2_B04', 's2_B03', 's2_B02']
NODATA = 65535


def _agri_year(t: pd.Timestamp) -> int:
    """Agricultural year yr spans Jul yr-1 .. Jun yr."""
    return t.year + 1 if t.month >= 7 else t.year


def _find_field_lnf(poly_id: int, t1: pd.Timestamp,
                    yr: int | None) -> tuple:
    """Field polygon by LNF ``id`` from lnf{yr}.gpkg (EPSG:2056 -> 32632)."""
    yr = int(yr) if yr is not None else _agri_year(t1)
    path = os.path.join(os.path.expanduser(LNF_DIR), f'lnf{yr}.gpkg')
    try:
        lnf = gpd.read_file(path, where=f'id = {int(poly_id)}')   # pyogrio: reads one row
    except TypeError:                                             # fiona engine: no `where`
        lnf = gpd.read_file(path)
        lnf = lnf[lnf['id'] == int(poly_id)]
    if lnf.empty:
        raise ValueError(f"LNF id={poly_id} not found in {path}. If this id comes "
                         "from samplesv2.pkl / the training outputs, use "
                         "id_source='samples'.")
    lnf = lnf.to_crs(S2_CRS)
    row = lnf.iloc[0]
    return row.geometry, row['lnf_code'], yr


def _find_field_samples(poly_id: int, t1: pd.Timestamp, t2: pd.Timestamp,
                        yr: int | None) -> tuple:
    """Field polygon by sample_FC poly_id (per-year LNF row index)."""
    samples = pd.read_pickle(SAMPLES_PATH)
    rows = samples[samples['poly_id'] == poly_id]
    if yr is not None:
        rows = rows[rows['yr'].astype(int) == int(yr)]
    if rows.empty:
        raise ValueError(f"poly_id={poly_id}" + (f", yr={yr}" if yr else "")
                         + f" not found in {SAMPLES_PATH}.")
    if len(rows) > 1:
        # Keep the year whose season window covers both timesteps
        def covers(y):
            y = int(y)
            lo, hi = pd.Timestamp(f'{y - 1}-06-01'), pd.Timestamp(f'{y}-07-30')
            return lo <= t1 <= hi and lo <= t2 <= hi
        in_season = rows[rows['yr'].apply(covers)]
        if len(in_season) == 1:
            rows = in_season
        else:
            raise ValueError(
                f"poly_id={poly_id} exists in years {sorted(rows['yr'].astype(int).unique())}; "
                "pass yr=... to pick one.")
    row = rows.iloc[0]
    polygon = gpd.GeoSeries([row['polygon_geom']], crs=samples.crs or S2_CRS) \
                 .to_crs(S2_CRS).iloc[0]
    return polygon, row['lnf_code'], int(row['yr'])


def _pick_tile(polygon, grid: gpd.GeoDataFrame) -> tuple[int, int]:
    """S2 tile (left, top) covering the largest part of the field."""
    hits = grid[grid.intersects(polygon)].copy()
    if hits.empty:
        raise ValueError("Field does not intersect any S2 tile in the grid.")
    hits['overlap'] = hits.geometry.intersection(polygon).area
    best = hits.sort_values('overlap', ascending=False).iloc[0]
    return int(best['left']), int(best['top'])


def _cloud_fraction(scene: xr.Dataset) -> float:
    """Fraction of cloud/shadow/snow pixels (same rules as clean_timeseries_field)."""
    bad = xr.zeros_like(scene[RGB_BANDS[0]], dtype=bool)
    if 's2_SCL' in scene:
        bad |= scene['s2_SCL'].isin([3, 8, 9, 10, 11])
    if 's2_mask' in scene:
        bad |= scene['s2_mask'].isin([1, 2, 3])
    return float(bad.mean())


def _select_scene(ds: xr.Dataset, t: pd.Timestamp, search_days: int) -> xr.Dataset:
    """Nearest acquisition to ``t``; with search_days > 0, the least cloudy one
    within ±search_days (ties broken by closeness to t)."""
    if search_days <= 0:
        return ds.sel(time=t, method='nearest')
    win = ds.sel(time=slice(t - pd.Timedelta(days=search_days),
                            t + pd.Timedelta(days=search_days)))
    if win.sizes['time'] == 0:
        return ds.sel(time=t, method='nearest')
    cols = [c for c in ('s2_SCL', 's2_mask') if c in win]
    masks = win[cols + [RGB_BANDS[0]]].load()
    scores = [(_cloud_fraction(masks.isel(time=i)),
               abs(pd.Timestamp(masks.time.values[i]) - t)) for i in range(win.sizes['time'])]
    best = min(range(len(scores)), key=lambda i: scores[i])
    return win.isel(time=best)


def _to_rgb(scene: xr.Dataset, lat_desc: bool) -> np.ndarray:
    rgb = np.stack([scene[b].values.astype('float32') for b in RGB_BANDS], axis=-1)
    rgb[(rgb == NODATA) | (rgb <= 0)] = np.nan
    rgb /= 10000.0                       # DN -> reflectance
    if not lat_desc:                     # make row 0 the northern edge
        rgb = rgb[::-1]
    return rgb


def plot_s2_rgb_pair(poly_id: int, timestep1, timestep2, yr: int | None = None,
                     id_source: str = 'lnf', pad_m: float = 150,
                     search_days: int = 0, save_path: str | None = None):
    """Plot two Sentinel-2 true-colour images of field ``poly_id`` side by side.

    Parameters
    ----------
    poly_id      : field id; meaning depends on ``id_source``.
    timestep1/2  : dates (str or Timestamp); the nearest S2 acquisition is used.
    yr           : optional agricultural year (Jul yr-1 .. Jun yr); inferred
                   from timestep1 if None.
    id_source    : 'lnf' -> poly_id is the LNF ``id`` (inference products);
                   'samples' -> poly_id from samplesv2.pkl (calibration/training).
    pad_m        : context margin around the field bounding box [m].
    search_days  : if > 0, take the least cloudy scene within ±search_days
                   instead of the strictly nearest one.
    save_path    : optional path to save the figure.
    """
    t1, t2 = pd.Timestamp(timestep1), pd.Timestamp(timestep2)

    if id_source == 'lnf':
        polygon, lnf_code, yr = _find_field_lnf(int(poly_id), t1, yr)
    elif id_source == 'samples':
        polygon, lnf_code, yr = _find_field_samples(int(poly_id), t1, t2, yr)
    else:
        raise ValueError(f"id_source must be 'lnf' or 'samples', got {id_source!r}")
    crs = S2_CRS

    grid = gpd.read_file(os.path.expanduser(S2_GRID_PATH)).to_crs(crs)
    left, top = _pick_tile(polygon, grid)

    s2_dir = os.path.expanduser(S2_DIR)
    years = sorted({t1.year, t2.year})
    files = sorted(os.path.join(s2_dir, f) for f in os.listdir(s2_dir)
                   if f.endswith('.zarr')
                   and f.startswith(tuple(f'S2_{left}_{top}_{y}' for y in years)))
    if not files:
        raise FileNotFoundError(f"No S2 files S2_{left}_{top}_{years}* in {s2_dir}")

    ds = xr.open_mfdataset(files, engine='zarr')

    # Crop to field bbox + margin (dims are lat/lon but hold UTM y/x)
    minx, miny, maxx, maxy = polygon.bounds
    minx, miny, maxx, maxy = minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m
    lat_desc = bool(ds.lat.values[0] > ds.lat.values[-1])
    ds = ds.sel(lon=slice(minx, maxx),
                lat=slice(maxy, miny) if lat_desc else slice(miny, maxy))
    bands = RGB_BANDS + [c for c in ('s2_SCL', 's2_mask') if c in ds]
    ds = ds[bands]

    scenes = [_select_scene(ds, t, search_days).load() for t in (t1, t2)]
    rgbs = [_to_rgb(s, lat_desc) for s in scenes]

    # Joint 2–98 % stretch so both panels are directly comparable
    valid = np.concatenate([r[np.isfinite(r).all(axis=-1)] for r in rgbs])
    lo, hi = (np.percentile(valid, [2, 98]) if len(valid) else (0.0, 0.3))

    res = float(abs(ds.lon.values[1] - ds.lon.values[0]))
    extent = [float(ds.lon.min()) - res / 2, float(ds.lon.max()) + res / 2,
              float(ds.lat.min()) - res / 2, float(ds.lat.max()) + res / 2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    outline = gpd.GeoSeries([polygon], crs=crs).boundary
    for ax, scene, rgb, t_req in zip(axes, scenes, rgbs, (t1, t2)):
        img = np.clip((rgb - lo) / (hi - lo), 0, 1)
        img = np.nan_to_num(img, nan=1.0)            # no-data shown white
        ax.imshow(img, extent=extent, origin='upper', interpolation='nearest')
        outline.plot(ax=ax, color='yellow', linewidth=1.5)
        t_img = pd.Timestamp(scene.time.values)
        ax.set_title(f"{t_img:%Y-%m-%d}  (requested {t_req:%Y-%m-%d})\n"
                     f"cloud/shadow/snow in view: {_cloud_fraction(scene):.0%}")
        ax.set_xlabel('Easting [m]')
        ax.ticklabel_format(style='plain', useOffset=False)
    axes[0].set_ylabel('Northing [m]')
    fig.suptitle(f"poly_id={poly_id} ({id_source})  yr={yr}  lnf_code={lnf_code}  "
                 f"tile S2_{left}_{top}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


if __name__ == '__main__':
    # p = argparse.ArgumentParser(description=__doc__,
    #                             formatter_class=argparse.RawDescriptionHelpFormatter)
    # p.add_argument('poly_id', type=int)
    # p.add_argument('timestep1')
    # p.add_argument('timestep2')
    # p.add_argument('--yr', type=int, default=None)
    # p.add_argument('--id-source', choices=['lnf', 'samples'], default='lnf')
    # p.add_argument('--search-days', type=int, default=0)
    # p.add_argument('--save', default=None)
    # a = p.parse_args()
    # plot_s2_rgb_pair(a.poly_id, a.timestep1, a.timestep2, yr=a.yr,
    #                  id_source=a.id_source, search_days=a.search_days,
    #                  save_path=a.save)

    poly_id = 12557
    timestep1 = "2020-10-09"
    timestep2 = "2020-10-19"
    year = 2020
    id_source = "lnf"
    search_days = 5
    save_path = "run_eb1/fc/rgb.png"
    plot_s2_rgb_pair(poly_id, timestep1, timestep2, yr=year, id_source=id_source,
                     search_days=search_days, save_path=save_path)