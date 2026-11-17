"""
Check for sudden, persistent geolocation shifts in Sentinel-1 RTC data
(Planetary Computer, collection "sentinel-1-rtc") at several Swiss sites.

For each site: compare every image to the mean of the first N images and plot
the average absolute VV difference over time. A sudden, persistent jump that
does not go away suggests a geolocation shift.
"""
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import xarray as xr
import numpy as np
import odc.stac
import math
import glob
import os
import re
import planetary_computer
import pystac_client
from pyproj import Transformer
from datetime import datetime
from scipy.ndimage import center_of_mass, label





# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_bbox(lat, lon, half_w_km, half_h_km):
    """Return [minx, miny, maxx, maxy] in degrees around a centre point."""
    dlat = half_h_km / 111.32
    dlon = half_w_km / (111.32 * np.cos(np.radians(lat)))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]

utm_transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:32632",
    always_xy=True,
)

def make_bbox_utm(lat, lon, half_km):

    x, y = utm_transformer.transform(lon, lat)

    d = half_km * 1000

    return [
        x - d,
        y - d,
        x + d,
        y + d,
    ]

def orbit_key(item):
    """Label like 'asc 15' or 'desc 66' from the item's SAR metadata."""
    p = item.properties
    state = p.get("sat:orbit_state", "?")
    state = {"ascending": "asc", "descending": "desc"}.get(state, state)
    return f"{state} {p.get('sat:relative_orbit', '?')}"


def compute_deltas(items, bbox, CRS="EPSG:32632", RESOLUTION=10, 
                   MIN_VALID_FRACTION=0.9, N_BASELINE=10):
    """Mean absolute VV difference of each image vs. the first-N baseline."""
    ds = odc.stac.load(
        items,
        bands=["vv"],
        bbox=bbox,
        crs=CRS,
        resolution=RESOLUTION,
    )
    vv = ds.vv.where(ds.vv > 0)  # linear power must be > 0; mask nodata

    # Drop scenes that only partly overlap the box (frame edges, masking)
    valid = vv.notnull().mean(dim=["x", "y"])
    vv = vv.isel(time=(valid >= MIN_VALID_FRACTION).values).sortby("time")

    if vv.sizes["time"] <= N_BASELINE:
        return None

    baseline = vv.isel(time=slice(None, N_BASELINE)).mean(dim="time")
    return np.abs(vv.isel(time=slice(N_BASELINE, None)) - baseline).mean(
        dim=["x", "y"]
    )

def largest_jump(deltas):
    """(jump size, date before, date after) of the largest step between
    consecutive acquisitions in one track's series."""
    v = deltas.values
    if len(v) < 2:
        return None
    steps = np.abs(np.diff(v))
    i = int(np.nanargmax(steps))
    return steps[i], deltas.time.values[i], deltas.time.values[i + 1]
 

def plot_s1_ts(time_sampling=False):
    # ---------------------------------------------------------------------------
    # Settings
    # ---------------------------------------------------------------------------
    SITES = {
        "Geneva (airport)": (46.2381, 6.1090),
        "Outside Zurich (Kloten airport)": (47.4582, 8.5555),
        "Samedan (Engadin airport)": (46.5341, 9.8841),
    }
    HALF_WIDTH_KM = 0.65
    HALF_HEIGHT_KM = 0.40
     
    DATETIME = "2023-01-01/2025-01-01"
    N_BASELINE = 10            # first N images of each track form the reference
    MIN_VALID_FRACTION = 0.9   # drop acquisitions that don't cover the box
    CRS = "EPSG:32632"
    RESOLUTION = 10
     
    # ---------------------------------------------------------------------------
    # Main
    # ---------------------------------------------------------------------------
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
     
    fig, axes = plt.subplots(len(SITES), 1, figsize=(11, 3.4 * len(SITES)), sharex=True)
    axes = np.atleast_1d(axes)
     
    for ax, (site, (lat, lon)) in zip(axes, SITES.items()):
        bbox = make_bbox(lat, lon, HALF_WIDTH_KM, HALF_HEIGHT_KM)
        items = catalog.search(
            collections=["sentinel-1-rtc"],
            bbox=bbox,
            datetime=DATETIME,
            query={
                "sar:instrument_mode": {"eq": "IW"},
                "sar:polarizations": {"eq": ["VV", "VH"]},
            },
        ).item_collection()
        print(f"{site}: {len(items)} items")
     
        groups = defaultdict(list)
        for item in items:
            groups[orbit_key(item)].append(item)
     
        best = None  # (jump, before, after, track) over all tracks of this site
        for label, group in sorted(groups.items()):
            deltas = compute_deltas(group, bbox)
            if deltas is None:
                print(f"  {label}: not enough valid images, skipped")
                continue
            ax.plot(deltas.time, deltas.values, ".", label=f"{label} (n={deltas.sizes['time']})")
            jump = largest_jump(deltas)
            if jump is not None and (best is None or jump[0] > best[0]):
                best = (*jump, label)
     
        if best is not None:
            size, t_before, t_after, track = best
            date_txt = pd.Timestamp(t_after).strftime("%Y-%m-%d")
            ax.axvline(t_after, color="red", lw=1, ls="--")
            ax.text(t_after, 0.97, date_txt, rotation=90, color="red", fontsize=9,
                    ha="right", va="top", transform=ax.get_xaxis_transform())
            print(f"  largest jump: {size:.4f} on track {track}, between "
                  f"{pd.Timestamp(t_before):%Y-%m-%d} and {date_txt}")
            ax.legend(fontsize=8, loc="upper left", ncol=3)
        else:
            ax.text(0.5, 0.5, "no usable data", transform=ax.transAxes, ha="center")
     
        ax.set_title(f"{site}: mean |VV - baseline of first {N_BASELINE} images|")
        ax.set_ylabel("Mean |ΔVV| (linear)")
     
    axes[-1].set_xlabel("Acquisition date")
    fig.tight_layout()
    fig.savefig("test/s1_rtc_shift_check_switzerland_2023.png", dpi=150)

def check_map():
    """
    Before/after maps of Sentinel-1 RTC (VV) over Zurich airport for one orbit
    track, with fixed reference lines along runway centrelines, to inspect a
    suspected geolocation shift around a given date.
    """
    # ---------------------------------------------------------------------------
    # Settings
    # ---------------------------------------------------------------------------
    SPLIT_DATE = "2023-11-01"     # suspected jump
    ORBIT_STATE = "ascending"
    RELATIVE_ORBIT = 15
    N_COMPOSITE = 5               # images averaged on each side (1 = single scenes)
    SEARCH_DAYS = 120             # how far to search either side of SPLIT_DATE
    
    # Map area around Zurich airport (lon/lat): [minx, miny, maxx, maxy]
    BBOX = [8.525, 47.443, 8.580, 47.482]
    CRS = "EPSG:32632"
    RESOLUTION = 10
    MIN_VALID_FRACTION = 0.9
    
    # Reference lines (lat, lon) along runway centrelines, from published runway
    # end coordinates (SkyVector). Runway concrete is dark in SAR, so the dark
    # stripe should sit on the line. A shift perpendicular to a line shows up as
    # the stripe moving off it. 10/28 runs E-W (reveals N-S shifts); 16/34 runs
    # NNW-SSE (reveals mostly E-W shifts). Remove one if you only want one line.
    REF_LINES = {
        "Runway 10/28": [(47.45900, 8.53750), (47.45667, 8.57050)],
        "Runway 16/34": [(47.47567, 8.53600), (47.44933, 8.55417)],
    }
    PROFILE_LINE = "Runway 10/28"  # transect drawn perpendicular to this line
    PROFILE_HALF_LENGTH_M = 300
    PROFILE_POSITION = 0.5         # where along that line (0 = first point, 1 = second)
    
    # ---------------------------------------------------------------------------
    # Search and load
    # ---------------------------------------------------------------------------
    split = pd.Timestamp(SPLIT_DATE)
    search_range = (
        f"{(split - pd.Timedelta(days=SEARCH_DAYS)).date()}/"
        f"{(split + pd.Timedelta(days=SEARCH_DAYS)).date()}"
    )
    
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    items = catalog.search(
        collections=["sentinel-1-rtc"],
        bbox=BBOX,
        datetime=search_range,
        query={
            "sar:instrument_mode": {"eq": "IW"},
            "sar:polarizations": {"eq": ["VV", "VH"]},
            "sat:orbit_state": {"eq": ORBIT_STATE},
            "sat:relative_orbit": {"eq": RELATIVE_ORBIT},
        },
    ).item_collection()
    print(f"{len(items)} items found for {ORBIT_STATE} {RELATIVE_ORBIT}")
    
    # Only one track is selected, so grouping by day just mosaics adjacent frames
    # of the same pass.
    ds = odc.stac.load(
        items, bands=["vv"], bbox=BBOX, crs=CRS, resolution=RESOLUTION,
        groupby="solar_day",
    )
    vv = ds.vv.where(ds.vv > 0)
    valid = vv.notnull().mean(dim=["x", "y"])
    vv = vv.isel(time=(valid >= MIN_VALID_FRACTION).values).sortby("time")
    
    before = vv.sel(time=vv.time < np.datetime64(split)).isel(time=slice(-N_COMPOSITE, None))
    after = vv.sel(time=vv.time >= np.datetime64(split)).isel(time=slice(None, N_COMPOSITE))
    if before.sizes["time"] == 0 or after.sizes["time"] == 0:
        raise SystemExit("No valid images on one side of the split date.")
    
    
    def to_db(x):
        return 10 * np.log10(x.median(dim="time"))
    
    
    def date_range(x):
        t = pd.to_datetime(x.time.values)
        return t[0].strftime("%Y-%m-%d") if len(t) == 1 else \
            f"{t[0]:%Y-%m-%d} to {t[-1]:%Y-%m-%d} (n={len(t)})"
    
    
    before_db, after_db = to_db(before), to_db(after)
    print("Before:", date_range(before))
    print("After: ", date_range(after))
    
    # ---------------------------------------------------------------------------
    # Reference lines and transect in map coordinates
    # ---------------------------------------------------------------------------
    to_utm = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    lines_utm = {
        name: np.array([to_utm.transform(lon, lat) for lat, lon in pts])
        for name, pts in REF_LINES.items()
    }
    
    p1, p2 = lines_utm[PROFILE_LINE]
    direction = (p2 - p1) / np.linalg.norm(p2 - p1)
    normal = np.array([-direction[1], direction[0]])
    mid = p1 + PROFILE_POSITION * (p2 - p1)
    s = np.arange(-PROFILE_HALF_LENGTH_M, PROFILE_HALF_LENGTH_M + 1, RESOLUTION / 2)
    tx, ty = mid[0] + s * normal[0], mid[1] + s * normal[1]
    
    
    def profile(img):
        return img.interp(
            x=xr.DataArray(tx, dims="s"), y=xr.DataArray(ty, dims="s")
        ).values
    
    
    # ---------------------------------------------------------------------------
    # Optional: estimate the overall shift by phase correlation
    # ---------------------------------------------------------------------------
    try:
        from skimage.registration import phase_cross_correlation
    
        a, b = before_db.values, after_db.values
        a = np.nan_to_num(a, nan=np.nanmedian(a))
        b = np.nan_to_num(b, nan=np.nanmedian(b))
        shift_px, _, _ = phase_cross_correlation(a, b, upsample_factor=20)
        dx_pix = float(before_db.x[1] - before_db.x[0])
        dy_pix = float(before_db.y[1] - before_db.y[0])
        east_m = -shift_px[1] * dx_pix
        north_m = -shift_px[0] * dy_pix
        shift_text = f"Estimated shift of 'after' vs 'before': {east_m:+.1f} m E, {north_m:+.1f} m N"
    except ImportError:
        shift_text = "Install scikit-image for an automatic shift estimate"
    print(shift_text)
    
    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    vmin, vmax = np.nanpercentile(before_db.values, [2, 98])
    colors = ["red", "cyan", "yellow", "magenta"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, img, label in [
        (axes[0], before_db, f"Before: {date_range(before)}"),
        (axes[1], after_db, f"After: {date_range(after)}"),
    ]:
        img.plot.imshow(ax=ax, cmap="gray", vmin=vmin, vmax=vmax, add_colorbar=False)
        for (name, xy), c in zip(lines_utm.items(), colors):
            ax.plot(xy[:, 0], xy[:, 1], color=c, lw=0.8, label=name)
        ax.plot(tx, ty, color="orange", lw=0.8, ls="--", label="Profile")
        ax.set_aspect("equal")
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_xlim([465750, 466250])
        ax.set_ylim([5256000,5256500])
    axes[0].legend(fontsize=8, loc="lower left")
    
    axes[2].plot(s, profile(before_db), label="Before")
    axes[2].plot(s, profile(after_db), label="After")
    axes[2].axvline(0, color="red", lw=0.8, label=f"{PROFILE_LINE} centreline")
    axes[2].set_xlabel(f"Distance across {PROFILE_LINE} (m, + = left of line direction)")
    axes[2].set_ylabel("VV (dB)")
    axes[2].set_title("Profile across runway", fontsize=10)
    axes[2].legend(fontsize=8)
    
    fig.suptitle(
        f"Sentinel-1 RTC VV, Zurich airport, {ORBIT_STATE} {RELATIVE_ORBIT}, split {SPLIT_DATE}\n{shift_text}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig("test/s1_rtc_shift_maps_zurich_zoom.png", dpi=150)



def check_map_local():
    """
    Before/after maps of local Sentinel-1 (VV) zarr tiles for a small zoom window
    at Zurich airport, plus SWISSIMAGE 10 cm true colour of the same window and
    a backscatter profile across a runway, to inspect a suspected geolocation
    shift around a given date.

    Only the tiles intersecting the zoom window are read.
    """
    # ---------------------------------------------------------------------------
    # Settings
    # ---------------------------------------------------------------------------
    SPLIT_DATE = "2023-11-01"     # suspected jump
    YEARS = [2023]
    GROUP = "asc"                 # zarr group in the local S1 files
    RELATIVE_ORBIT = 15
    PLATFORM = None               # None = all, or "A" / "B" to use one satellite
    N_COMPOSITE = 5               # images averaged on each side (1 = single scenes)
    SEARCH_DAYS = 120             # how far to search either side of SPLIT_DATE
    MIN_VALID_FRACTION = 0.9

    # Analysis window in EPSG:32632 (the only area that is loaded)
    ZOOM_X = [465750, 466250]
    ZOOM_Y = [5256000, 5256500]
    CRS = "EPSG:32632"
    TILE_M = 1280                 # S1 and SWISSIMAGE tiles: 1280 m, named by x_left / y_top

    # SWISSIMAGE 10 cm tiles: <SWISSIMAGE_DIR>/SwissImage*_<x_left>_<y_top>.zarr
    SWISSIMAGE_DIR = "/mnt/eo-nas1/data/swisstopo/SwissImage/cubes/10cm"      # <- adjust
    SWISSIMAGE_GROUP = None                    # zarr group, if the files use one
    SWISSIMAGE_PLOT_RES = 0.5                  # m; block-average 10 cm to this for plotting

    REF_LINES = {
        "Runway 10/28": [(47.45900, 8.53750), (47.45667, 8.57050)],
        "Runway 16/34": [(47.47567, 8.53600), (47.44933, 8.55417)],
    }
    PROFILE_LINE = "Runway 10/28"
    PROFILE_HALF_LENGTH_M = 200   # must fit into the zoom window

    OUT_PNG = f"test/fola/s1_local_shift_maps_zurich_{GROUP}{RELATIVE_ORBIT}_zoom_swissimage.png"

    # ---------------------------------------------------------------------------
    # Geometry: window, reference lines, profile
    # ---------------------------------------------------------------------------
    split = pd.Timestamp(SPLIT_DATE)
    t_start = split - pd.Timedelta(days=SEARCH_DAYS)
    t_end = split + pd.Timedelta(days=SEARCH_DAYS)

    to_utm = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    to_ll = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)

    xmin, xmax = ZOOM_X
    ymin, ymax = ZOOM_Y
    xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2

    lines_utm = {
        name: np.array([to_utm.transform(lon, lat) for lat, lon in pts])
        for name, pts in REF_LINES.items()
    }

    # profile centred on the point of PROFILE_LINE closest to the window centre
    p1, p2 = lines_utm[PROFILE_LINE]
    length = np.linalg.norm(p2 - p1)
    direction = (p2 - p1) / length
    normal = np.array([-direction[1], direction[0]])
    along = np.clip(np.dot(np.array([xc, yc]) - p1, direction), 0, length)
    mid = p1 + along * direction
    s = np.arange(-PROFILE_HALF_LENGTH_M, PROFILE_HALF_LENGTH_M + 1, 5)
    tx, ty = mid[0] + s * normal[0], mid[1] + s * normal[1]
    if not (xmin <= mid[0] <= xmax and ymin <= mid[1] <= ymax):
        print(f"WARNING: {PROFILE_LINE} does not cross the zoom window; "
              "the profile will be empty.")

    # ---------------------------------------------------------------------------
    # Find S1 tiles for the window
    # ---------------------------------------------------------------------------
    def parse_s1(f):
        """S1_<x_left>_<y_top>_<YYYYMMDD>_<YYYYMMDD>.zarr -> (x_left, y_top, years)"""
        m = re.search(r"S1_(\d+)_(\d+)_(\d{8})_(\d{8})\.zarr",
                      os.path.basename(str(f).rstrip("/")))
        if m is None:
            return None
        return int(m[1]), int(m[2]), set(range(int(m[3][:4]), int(m[4][:4]) + 1))

    lon_c, lat_c = to_ll.transform(xc, yc)
    first = next((i for i in map(parse_s1, find_tiles(lat_c, lon_c)) if i), None)
    if first is None:
        raise SystemExit("find_tiles returned no S1 tile for the window centre.")
    anchor_x, anchor_y = first[0], first[1]

    # all 1280 m tiles intersecting the window (tile = x_left..+1280, y_top-1280..y_top)
    kx = np.arange(np.floor((xmin - anchor_x) / TILE_M), np.floor((xmax - anchor_x) / TILE_M) + 1)
    ky = np.arange(np.ceil((ymin - anchor_y) / TILE_M), np.ceil((ymax - anchor_y) / TILE_M) + 1)
    expected = {(int(anchor_x + i * TILE_M), int(anchor_y + j * TILE_M)) for i in kx for j in ky}

    s1_files = {}
    for xl, yt in sorted(expected):
        lon, lat = to_ll.transform(xl + TILE_M / 2, yt - TILE_M / 2)
        for f in find_tiles(lat, lon):
            info = parse_s1(f)
            if info and (info[0], info[1]) == (xl, yt) and info[2] & set(YEARS):
                paths = s1_files.setdefault((xl, yt), [])
                if f not in paths:
                    paths.append(f)

    print(f"S1 tiles: {len(expected)} needed, {len(s1_files)} found")
    for k in sorted(expected - set(s1_files)):
        print(f"  MISSING S1_{k[0]}_{k[1]}_...")
    if not s1_files:
        raise SystemExit(f"No S1 files found for years {YEARS}.")

    # ---------------------------------------------------------------------------
    # Load S1 tiles (time / orbit / platform filter, crop to window)
    # ---------------------------------------------------------------------------
    def crop(obj, xdim, ydim):
        xs, ys = obj[xdim].values, obj[ydim].values
        return obj.sel({
            xdim: slice(xmin, xmax) if xs[0] < xs[-1] else slice(xmax, xmin),
            ydim: slice(ymax, ymin) if ys[0] > ys[-1] else slice(ymin, ymax),
        })

    pieces = []
    for key, paths in sorted(s1_files.items()):
        for f in paths:
            z = xr.open_zarr(f, group=GROUP)
            t = pd.to_datetime(z.time.values)
            m = (
                (z.relative_orbit.compute().values == RELATIVE_ORBIT)
                & (t >= t_start) & (t <= t_end)
            )
            if PLATFORM is not None:
                plat = np.char.upper(np.char.strip(
                    np.asarray(z.platform.compute().values).astype(str)
                ))
                m &= np.char.endswith(plat, PLATFORM.upper())
            if not m.any():
                continue
            z = crop(z.isel(time=m), "lon", "lat")
            if z.sizes["lon"] == 0 or z.sizes["lat"] == 0:
                continue

            da = z.s1_vv.where(z.s1_valid > 0)
            da = da.where(da > 0).rename(lon="x", lat="y").transpose("time", "y", "x")
            da = da.assign_coords(time=da.time.dt.floor("D")).compute()
            if not da.indexes["time"].is_unique:      # two frames on the same day
                da = da.groupby("time").mean(skipna=True)
            pieces.append(da)

    if not pieces:
        raise SystemExit(f"No acquisitions for {GROUP} {RELATIVE_ORBIT} in the time window.")

    # ---------------------------------------------------------------------------
    # Mosaic S1 onto one regular grid (north-up, missing tiles = NaN)
    # ---------------------------------------------------------------------------
    RESOLUTION = abs(float(np.median(np.diff(pieces[0].x.values))))
    gx = float(pieces[0].x.values[0])     # keeps the tiles' own pixel grid
    gy = float(pieces[0].y.values[0])
    x_reg = gx + RESOLUTION * np.arange(
        np.ceil((xmin - gx) / RESOLUTION), np.floor((xmax - gx) / RESOLUTION) + 1
    )
    y_reg = gy + RESOLUTION * np.arange(
        np.floor((ymax - gy) / RESOLUTION), np.ceil((ymin - gy) / RESOLUTION) - 1, -1
    )
    times = pd.DatetimeIndex(sorted(set(np.concatenate([p.time.values for p in pieces]))))

    acc = np.zeros((len(times), len(y_reg), len(x_reg)))
    cnt = np.zeros_like(acc, dtype="int16")
    for p in pieces:
        it = times.get_indexer(pd.to_datetime(p.time.values))
        ix = np.round((p.x.values - x_reg[0]) / RESOLUTION).astype(int)
        iy = np.round((y_reg[0] - p.y.values) / RESOLUTION).astype(int)
        ok_x = (ix >= 0) & (ix < len(x_reg))
        ok_y = (iy >= 0) & (iy < len(y_reg))
        v = p.values[:, ok_y][:, :, ok_x]
        idx = np.ix_(it, iy[ok_y], ix[ok_x])
        acc[idx] += np.nan_to_num(v)
        cnt[idx] += np.isfinite(v)

    mosaic = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    vv = xr.DataArray(mosaic, dims=("time", "y", "x"),
                      coords={"time": times, "y": y_reg, "x": x_reg})
    print(f"S1 mosaic: {len(times)} dates, {len(x_reg)} x {len(y_reg)} px")

    # ---------------------------------------------------------------------------
    # Select valid scenes and build composites
    # ---------------------------------------------------------------------------
    covered = vv.notnull().any(dim="time")
    valid = vv.notnull().where(covered).mean(dim=["x", "y"])

    print("\nValid fraction per acquisition:")
    for t, v in zip(pd.to_datetime(vv.time.values), valid.values):
        side = "before" if t < split else "after "
        flag = "" if v >= MIN_VALID_FRACTION else "  <- dropped"
        print(f"  {t:%Y-%m-%d}  {side}  {v:.2f}{flag}")

    vv = vv.isel(time=(valid >= MIN_VALID_FRACTION).values).sortby("time")

    before = vv.sel(time=vv.time < np.datetime64(split)).isel(time=slice(-N_COMPOSITE, None))
    after = vv.sel(time=vv.time >= np.datetime64(split)).isel(time=slice(None, N_COMPOSITE))
    print(f"\n{before.sizes['time']} before, {after.sizes['time']} after split")
    if before.sizes["time"] == 0 or after.sizes["time"] == 0:
        raise SystemExit("No valid images on one side of the split date.")

    def to_db(x):
        return 10 * np.log10(x.median(dim="time"))

    def date_range(x):
        t = pd.to_datetime(x.time.values)
        return t[0].strftime("%Y-%m-%d") if len(t) == 1 else \
            f"{t[0]:%Y-%m-%d} to {t[-1]:%Y-%m-%d} (n={len(t)})"

    before_db, after_db = to_db(before), to_db(after)
    print("Before:", date_range(before))
    print("After: ", date_range(after))

    def profile(img):
        return img.interp(
            x=xr.DataArray(tx, dims="s"), y=xr.DataArray(ty, dims="s")
        ).values

    # ---------------------------------------------------------------------------
    # Shift estimate by phase correlation (small window -> treat as indicative)
    # ---------------------------------------------------------------------------
    try:
        from skimage.registration import phase_cross_correlation

        a, b = before_db.values, after_db.values
        a = np.nan_to_num(a, nan=np.nanmedian(a))
        b = np.nan_to_num(b, nan=np.nanmedian(b))
        shift_px, _, _ = phase_cross_correlation(a, b, upsample_factor=20)
        dx_pix = float(before_db.x[1] - before_db.x[0])
        dy_pix = float(before_db.y[1] - before_db.y[0])
        east_m = -shift_px[1] * dx_pix
        north_m = -shift_px[0] * dy_pix
        shift_text = f"Estimated shift of 'after' vs 'before': {east_m:+.1f} m E, {north_m:+.1f} m N"
    except ImportError:
        shift_text = "Install scikit-image for an automatic shift estimate"
    print(shift_text)

    # ---------------------------------------------------------------------------
    # SWISSIMAGE 10 cm for the window
    # ---------------------------------------------------------------------------
    def parse_swiss(f):
        m = re.search(r"SwissImage.*?_(\d+)_(\d+)\.zarr$", os.path.basename(str(f).rstrip("/")))
        return None if m is None else (int(m[1]), int(m[2]))

    def load_rgb(path):
        """Open a SWISSIMAGE zarr -> DataArray (band, y, x)."""
        ds = xr.open_zarr(path, group=SWISSIMAGE_GROUP) if SWISSIMAGE_GROUP \
            else xr.open_zarr(path)
        for cand in (("red", "green", "blue"), ("R", "G", "B"), ("r", "g", "b")):
            if all(c in ds.data_vars for c in cand):
                da = xr.concat([ds[c] for c in cand], dim="band")
                break
        else:
            da = ds[list(ds.data_vars)[0]]
            band_dim = next(d for d in da.dims if d in ("band", "bands", "channel"))
            da = da.rename({band_dim: "band"}).isel(band=slice(0, 3))
        rename = {d: n for d, n in (("lon", "x"), ("lat", "y")) if d in da.dims}
        return da.rename(rename)

    swiss_files = []
    for f in sorted(glob.glob(os.path.join(SWISSIMAGE_DIR, "SwissImage*.zarr"))):
        k = parse_swiss(f)
        if k and k[0] < xmax and k[0] + TILE_M > xmin and k[1] > ymin and k[1] - TILE_M < ymax:
            swiss_files.append(f)
    print(f"\nSWISSIMAGE tiles: {len(swiss_files)} "
          f"{[os.path.basename(f) for f in swiss_files]}")

    swiss = None
    if swiss_files:
        r = SWISSIMAGE_PLOT_RES
        sx = np.arange(xmin + r / 2, xmax, r)
        sy = np.arange(ymax - r / 2, ymin, -r)
        parts = []
        for f in swiss_files:
            da = crop(load_rgb(f), "x", "y")
            if da.sizes["x"] == 0 or da.sizes["y"] == 0:
                continue
            res = abs(float(np.median(np.diff(da.x.values))))
            k = max(1, int(round(r / res)))
            da = da.astype("float32").coarsen(x=k, y=k, boundary="trim").mean()
            da = da.compute().sortby("x").sortby("y")
            parts.append(da.reindex(x=sx, y=sy, method="nearest", tolerance=r))

        if parts:
            sw = xr.concat(parts, dim="tile").mean("tile", skipna=True)
            sw = sw.transpose("y", "x", "band").values
            scale = 255.0 if np.nanmax(sw) > 1 else 1.0
            swiss = np.clip(sw / scale, 0, 1)
            swiss = np.where(np.isfinite(swiss), swiss, 1.0)   # gaps white

    # ---------------------------------------------------------------------------
    # Plot: before | after | SWISSIMAGE | profile
    # ---------------------------------------------------------------------------
    vmin, vmax = np.nanpercentile(before_db.values, [2, 98])
    colors = ["red", "cyan", "yellow", "magenta"]
    track = f"{GROUP}-{RELATIVE_ORBIT}" + (f" S1{PLATFORM}" if PLATFORM else "")

    fig, axes = plt.subplots(
        1, 4, figsize=(24, 6), gridspec_kw={"width_ratios": [1, 1, 1, 0.9]}
    )

    def decorate(ax, title):
        for (name, xy), c in zip(lines_utm.items(), colors):
            ax.plot(xy[:, 0], xy[:, 1], color=c, lw=0.8, label=name)
        ax.plot(tx, ty, color="orange", lw=0.8, ls="--", label="Profile")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")

    for ax, img, label in [
        (axes[0], before_db, f"Before: {date_range(before)}"),
        (axes[1], after_db, f"After: {date_range(after)}"),
    ]:
        img.plot.imshow(ax=ax, cmap="gray", vmin=vmin, vmax=vmax, add_colorbar=False)
        decorate(ax, label)
    axes[0].legend(fontsize=8, loc="lower left")

    if swiss is not None:
        axes[2].imshow(swiss, extent=[xmin, xmax, ymin, ymax], origin="upper")
        decorate(axes[2], f"SWISSIMAGE 10 cm (shown at {SWISSIMAGE_PLOT_RES:g} m)")
    else:
        axes[2].text(0.5, 0.5, "No SWISSIMAGE tile found", ha="center", va="center",
                     transform=axes[2].transAxes)
        decorate(axes[2], "SWISSIMAGE 10 cm")

    axes[3].plot(s, profile(before_db), label="Before")
    axes[3].plot(s, profile(after_db), label="After")
    axes[3].axvline(0, color="red", lw=0.8, label=f"{PROFILE_LINE} centreline")
    axes[3].set_xlabel(f"Distance across {PROFILE_LINE} (m, + = left of line direction)")
    axes[3].set_ylabel("VV (dB)")
    axes[3].set_title("Profile across runway", fontsize=10)
    axes[3].legend(fontsize=8)

    fig.suptitle(
        f"Sentinel-1 VV (local), Zurich airport, {track}, split {SPLIT_DATE}\n{shift_text}",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)


def check_geneva_desc():
    SITES = {
        "Geneva (airport)": (46.2381, 6.1090),
        "Outside Zurich (Kloten airport)": (47.4582, 8.5555),
        "Samedan (Engadin airport)": (46.5341, 9.8841),
        "Bellinzona": (46.1950, 9.0220),
    }
    HALF_WIDTH_KM = 0.65
    HALF_HEIGHT_KM = 0.40
    DATETIME = "2018-01-01/2026-01-01"
    CRS = "EPSG:32632"         # UTM 32N covers all four sites
    RESOLUTION = 10            # metres
    
    lat, lon = SITES["Geneva (airport)"]
    bbox = make_bbox(lat, lon, HALF_WIDTH_KM, HALF_HEIGHT_KM)

    catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
    )
    
    items = catalog.search(
        collections=["sentinel-1-rtc"], bbox=bbox, datetime=DATETIME,
        query={"sat:orbit_state": {"eq": "descending"}, "sat:relative_orbit": {"eq": 139}},
    ).item_collection()
    
    # 1) How many items per year, and with which polarisations?
    df = pd.DataFrame({"date": [it.datetime for it in items],
                       "pol": [tuple(it.properties["sar:polarizations"]) for it in items]})
    print(df.groupby([df.date.dt.year, "pol"]).size())
    
    # 2) How much of the box does each item from 2021 on actually cover?
    late = [it for it in items if it.datetime.year >= 2021]
    ds = odc.stac.load(late, bands=["vv"], bbox=bbox, crs=CRS, resolution=RESOLUTION)
    print(ds.vv.where(ds.vv > 0).notnull().mean(dim=["x", "y"]).to_series())

def plot_spatial3locs():
    """
    Zoomed before/after maps (60 x 60 pixels) of Sentinel-1 RTC VV at three Swiss
    airports, with per-site orbit track and dates, fixed runway reference lines
    and a profile across the runway.
    
    For each site you set:
      orbit  : the track to use, e.g. ("ascending", 15)
      before : the "before" image is the LAST valid acquisition on or before this date
      after  : the "after" image is the FIRST valid acquisition on or after this date
    The jump script (s1_rtc_shift_check_jumps.py) prints the track and the two
    dates around the largest jump; copy them here to see exactly those scenes.
    """
    from collections import defaultdict
    
    import matplotlib.pyplot as plt
    import numpy as np
    import odc.stac
    import pandas as pd
    import planetary_computer
    import pystac_client
    import xarray as xr
    from matplotlib.ticker import MaxNLocator
    from pyproj import Transformer
    
    # ---------------------------------------------------------------------------
    # Settings per site  (edit orbit / before / after here)
    # ---------------------------------------------------------------------------
    
    SITES = {
        "Zurich": {
            "orbit": ("ascending", 15),
            "before": "2023-09-01",
            "after": "2023-09-25",
            # Runway end coordinates from SkyVector
            "lines": {
                "Runway 10/28": [(47.45900, 8.53750), (47.45667, 8.57050)],
                "Runway 16/34": [(47.47567, 8.53600), (47.44933, 8.55417)],
            },
            "center": None,  # (lat, lon) of the window; None = middle of first line
        },
        "Geneva": {
            "orbit": ("ascending", 88),     # TODO: set from the jump script
            "before": "2024-05-27",         # TODO
            "after": "2024-06-08",          # TODO
            # Runway 04/22 threshold coordinates (sprinkle.com airport data)
            "lines": {"Runway 04/22": [(46.2258, 6.0909), (46.2504, 6.1270)]},
            "center": None,
        },
        "Samedan": {
            "orbit": ("ascending", 15),     # TODO: set from the jump script
            "before": "2023-08-25",         # TODO
            "after": "2024-09-01",          # TODO
            # Approximate: runway 03 end extended 1842 m along true bearing 29.15 deg
            "lines": {"Runway 03/21": [(46.52683, 9.87829), (46.54128, 9.89000)]},
            "center": None,
        },
    }
    
    WINDOW_PX = 60                 # map size in pixels (each side)
    RESOLUTION = 10                # metres per pixel
    CRS = "EPSG:32632"
    SEARCH_DAYS = 90               # how far before/after the dates to look for scenes
    MIN_VALID_FRACTION = 0.95
    PROFILE_HALF_LENGTH_M = 250    # must fit inside the 600 m window
    
    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    to_utm = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    to_ll = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    HALF = WINDOW_PX * RESOLUTION / 2
    
    
    def window_bbox_ll(cx, cy, margin=100):
        """Lon/lat bbox covering the UTM window plus a small margin."""
        xs = [cx - HALF - margin, cx + HALF + margin]
        ys = [cy - HALF - margin, cy + HALF + margin]
        lons, lats = to_ll.transform([xs[0], xs[1], xs[0], xs[1]], [ys[0], ys[0], ys[1], ys[1]])
        return [min(lons), min(lats), max(lons), max(lats)]
    
    
    def crop(img, cx, cy):
        y_desc = img.y.values[0] > img.y.values[-1]
        yslice = slice(cy + HALF, cy - HALF) if y_desc else slice(cy - HALF, cy + HALF)
        return img.sel(x=slice(cx - HALF, cx + HALF), y=yslice)
    
    
    def load_date(items, bbox, cx, cy):
        """Load all frames of one pass, crop to the window, return dB or None."""
        ds = odc.stac.load(items, bands=["vv"], bbox=bbox, crs=CRS,
                           resolution=RESOLUTION, groupby="solar_day")
        vv = crop(ds.vv.isel(time=0), cx, cy)
        vv = vv.where(vv > 0)
        if float(vv.notnull().mean()) < MIN_VALID_FRACTION:
            return None
        return 10 * np.log10(vv)
    
    
    def pick(by_date, dates, bbox, cx, cy):
        for d in dates:
            img = load_date(by_date[d], bbox, cx, cy)
            if img is not None:
                return img, d
        return None, None
    
    
    def estimate_shift(a, b):
        try:
            from skimage.registration import phase_cross_correlation
        except ImportError:
            return None
        av = np.nan_to_num(a.values, nan=np.nanmedian(a.values))
        bv = np.nan_to_num(b.values, nan=np.nanmedian(b.values))
        shift_px, _, _ = phase_cross_correlation(av, bv, upsample_factor=20)
        return (-shift_px[1] * float(a.x[1] - a.x[0]),
                -shift_px[0] * float(a.y[1] - a.y[0]))
    
    
    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    line_colors = ["red", "cyan", "yellow", "magenta"]
    fig, axes = plt.subplots(len(SITES), 3, figsize=(17, 5.6 * len(SITES)),
                             gridspec_kw={"width_ratios": [1, 1, 0.9]}, squeeze=False)
    
    for row, (name, site) in zip(axes, SITES.items()):
        lines = {ln: np.array([to_utm.transform(lon, lat) for lat, lon in pts])
                 for ln, pts in site["lines"].items()}
        p1, p2 = next(iter(lines.values()))
        profile_name = next(iter(lines))
        if site["center"] is None:
            cx, cy = (p1 + p2) / 2
        else:
            cx, cy = to_utm.transform(site["center"][1], site["center"][0])
        bbox = window_bbox_ll(cx, cy)
    
        before_day = pd.Timestamp(site["before"]).date()
        after_day = pd.Timestamp(site["after"]).date()
        start = (pd.Timestamp(before_day) - pd.Timedelta(days=SEARCH_DAYS)).date()
        end = (pd.Timestamp(after_day) + pd.Timedelta(days=SEARCH_DAYS)).date()
    
        state, rel = site["orbit"]
        items = catalog.search(
            collections=["sentinel-1-rtc"], bbox=bbox, datetime=f"{start}/{end}",
            query={"sar:instrument_mode": {"eq": "IW"},
                   "sar:polarizations": {"eq": ["VV", "VH"]}},
        ).item_collection()
        on_track = [it for it in items
                    if (it.properties.get("sat:orbit_state"), it.properties.get("sat:relative_orbit")) == (state, rel)]
        orbit_txt = f"{state} {rel}"
    
        by_date = defaultdict(list)
        for it in on_track:
            by_date[it.datetime.date()].append(it)
        before_img, b_date = pick(by_date, sorted([d for d in by_date if d <= before_day], reverse=True), bbox, cx, cy)
        after_img, a_date = pick(by_date, sorted([d for d in by_date if d >= after_day]), bbox, cx, cy)
    
        if before_img is None or after_img is None:
            tracks = sorted({(it.properties.get("sat:orbit_state"), it.properties.get("sat:relative_orbit")) for it in items})
            msg = f"{name}: no valid {'before' if before_img is None else 'after'} scene on {orbit_txt}"
            print(msg + f"\n  tracks available here: {tracks}")
            for ax in row:
                ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", wrap=True)
            continue
        print(f"{name}: {orbit_txt}, before {b_date}, after {a_date}")
    
        # Profile perpendicular to the first line, through the point on that line
        # closest to the window centre
        d = (p2 - p1) / np.linalg.norm(p2 - p1)
        n = np.array([-d[1], d[0]])
        foot = p1 + np.dot(np.array([cx, cy]) - p1, d) * d
        s = np.arange(-PROFILE_HALF_LENGTH_M, PROFILE_HALF_LENGTH_M + 1, RESOLUTION / 2)
        tx, ty = foot[0] + s * n[0], foot[1] + s * n[1]
    
        imgs = [before_img, after_img]
        labels = [f"Before: {b_date}", f"After: {a_date}"]
        vmin, vmax = np.nanpercentile(np.concatenate([i.values.ravel() for i in imgs]), [2, 98])
    
        for ax, img, lab in zip(row[:2], imgs, labels):
            img.plot.imshow(ax=ax, cmap="gray", vmin=vmin, vmax=vmax,
                            add_colorbar=False, interpolation="nearest")
            for (ln, xy), c in zip(lines.items(), line_colors):
                ax.plot(xy[:, 0], xy[:, 1], color=c, lw=1, label=ln)
            ax.plot(tx, ty, color="orange", lw=1, ls="--", label="Profile")
            ax.set_xlim(cx - HALF, cx + HALF)
            ax.set_ylim(cy - HALF, cy + HALF)
            ax.set_aspect("equal")
            ax.ticklabel_format(style="plain", useOffset=False)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.set_title(f"{name} {lab} ({orbit_txt})", fontsize=10)
            ax.set_xlabel("Easting (m)")
            ax.set_ylabel("Northing (m)")
        row[0].legend(fontsize=7, loc="lower left")
    
        for img, lab in zip(imgs, labels):
            prof = img.interp(x=xr.DataArray(tx, dims="s"), y=xr.DataArray(ty, dims="s"))
            row[2].plot(s, prof.values, marker=".", ms=3, label=lab)
        row[2].axvline(0, color="red", lw=0.8, label=f"{profile_name} centreline")
        row[2].set_xlabel(f"Distance across {profile_name} (m)")
        row[2].set_ylabel("VV (dB)")
    
        shift = estimate_shift(before_img, after_img)
        shift_txt = (f"after vs before: {shift[0]:+.1f} m E, {shift[1]:+.1f} m N"
                     if shift is not None else "install scikit-image for a shift estimate")
        print(f"  {shift_txt}")
        row[2].set_title(f"Profile across runway\n{shift_txt}", fontsize=10)
        row[2].legend(fontsize=8)
    
    fig.suptitle(f"Sentinel-1 RTC VV, {WINDOW_PX} x {WINDOW_PX} pixel windows", fontsize=12)
    fig.tight_layout()
    fig.savefig("test/s1_rtc_shift_maps_airports_2.png", dpi=150)

# """
# Check for sudden, persistent geolocation shifts in Sentinel-1 RTC data
# (Planetary Computer, "sentinel-1-rtc") at stable reference structures in
# Switzerland.

# For each site and orbit track: compare every image to the mean of the first
# N images and plot the average absolute VV difference over time. The largest
# jump between two consecutive acquisitions of the same track (any track) is
# marked with a vertical line labelled with its date.

# Call plot_shift_timeseries(full_timeseries=True) for 2016-2025 thinned out in
# time, or full_timeseries=False for 2023-2024 with every acquisition.
# """
# from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import pystac_client

# ---------------------------------------------------------------------------
# Reference sites: (lat, lon) centre and half size of the box in km
# ---------------------------------------------------------------------------
SITES = {
    "Zurich airport terminal": {"center": (47.4506, 8.5617), "half_km": 0.4},
    "Palexpo, Geneva": {"center": (46.2349, 6.1182), "half_km": 0.4},
    "Verzasca (Contra) dam": {"center": (46.1967, 8.8478), "half_km": 0.4},
}

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
FULL_PERIOD = "2016-01-01/2026-01-01"
SHORT_PERIOD = "2023-01-01/2025-01-01"
SAMPLE_EVERY = 20              # thinning used only in full mode ...
SAMPLE_UNIT = "days"           # ... "days" (min. spacing per track) or "acquisitions"

N_BASELINE = 10                # first N images of each track form the reference
MIN_VALID_FRACTION = 0.9       # drop acquisitions that don't cover the box
CRS = "EPSG:32632"
RESOLUTION = 10

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_bbox(lat, lon, half_km):
    dlat = half_km / 111.32
    dlon = half_km / (111.32 * np.cos(np.radians(lat)))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def orbit_key(item):
    p = item.properties
    state = {"ascending": "asc", "descending": "desc"}.get(
        p.get("sat:orbit_state"), p.get("sat:orbit_state", "?"))
    return f"{state} {p.get('sat:relative_orbit', '?')}"


def thin_dates(dates, every, unit):
    """Subsample a sorted list of dates."""
    if unit == "acquisitions":
        return dates[::every]
    kept = []
    for d in dates:
        if not kept or (d - kept[-1]).days >= every:
            kept.append(d)
    return kept


def compute_deltas(items, bbox):
    """Mean absolute VV difference of each image vs. the first-N baseline."""
    # Items belong to one track, so grouping by day only mosaics adjacent
    # frames of the same pass.
    ds = odc.stac.load(items, bands=["vv"], bbox=bbox, crs=CRS,
                       resolution=RESOLUTION, groupby="solar_day")
    vv = ds.vv.where(ds.vv > 0)
    valid = vv.notnull().mean(dim=["x", "y"])
    vv = vv.isel(time=(valid >= MIN_VALID_FRACTION).values).sortby("time")
    if vv.sizes["time"] <= N_BASELINE:
        return None
    baseline = vv.isel(time=slice(None, N_BASELINE)).mean(dim="time")
    return np.abs(vv.isel(time=slice(N_BASELINE, None)) - baseline).mean(dim=["x", "y"])


def largest_jump(deltas):
    v = deltas.values
    if len(v) < 2:
        return None
    steps = np.abs(np.diff(v))
    i = int(np.nanargmax(steps))
    return steps[i], deltas.time.values[i], deltas.time.values[i + 1]


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def plot_shift_timeseries(full_timeseries=False, sites=SITES):
    """
    full_timeseries=True : 2016-2025, thinned to SAMPLE_EVERY SAMPLE_UNIT per track
    full_timeseries=False: 2023-2024, every acquisition
    """
    period = FULL_PERIOD if full_timeseries else SHORT_PERIOD
    mode_txt = (f"{period}, every {SAMPLE_EVERY} {SAMPLE_UNIT}" if full_timeseries
                else f"{period}, all acquisitions")

    fig, axes = plt.subplots(len(sites), 1, figsize=(12, 3.2 * len(sites)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, (site, cfg) in zip(axes, sites.items()):
        lat, lon = cfg["center"]
        bbox = make_bbox(lat, lon, cfg.get("half_km", 0.4))
        items = catalog.search(
            collections=["sentinel-1-rtc"],
            bbox=bbox,
            datetime=period,
            query={
                "sar:instrument_mode": {"eq": "IW"},
                "sar:polarizations": {"eq": ["VV", "VH"]},
            },
        ).item_collection()
        print(f"{site}: {len(items)} items")

        groups = defaultdict(list)
        for item in items:
            groups[orbit_key(item)].append(item)

        best = None
        for label, group in sorted(groups.items()):
            if full_timeseries:
                by_date = defaultdict(list)
                for it in group:
                    by_date[it.datetime.date()].append(it)
                kept = thin_dates(sorted(by_date), SAMPLE_EVERY, SAMPLE_UNIT)
                group = [it for d in kept for it in by_date[d]]

            deltas = compute_deltas(group, bbox)
            if deltas is None:
                print(f"  {label}: not enough valid images, skipped")
                continue
            ax.plot(deltas.time, deltas.values, ".", label=f"{label} (n={deltas.sizes['time']})")
            jump = largest_jump(deltas)
            if jump is not None and (best is None or jump[0] > best[0]):
                best = (*jump, label)

        if best is not None:
            size, t_before, t_after, track = best
            date_txt = pd.Timestamp(t_after).strftime("%Y-%m-%d")
            ax.axvline(t_after, color="red", lw=1, ls="--")
            ax.text(t_after, 0.97, date_txt, rotation=90, color="red", fontsize=9,
                    ha="right", va="top", transform=ax.get_xaxis_transform())
            print(f"  largest jump: {size:.4f} on track {track}, between "
                  f"{pd.Timestamp(t_before):%Y-%m-%d} and {date_txt}")
            ax.legend(fontsize=7, loc="upper left", ncol=4)
        else:
            ax.text(0.5, 0.5, "no usable data", transform=ax.transAxes, ha="center")

        ax.set_title(f"{site}: mean |VV - baseline of first {N_BASELINE} images|", fontsize=10)
        ax.set_ylabel("Mean |ΔVV| (linear)")

    axes[-1].set_xlabel("Acquisition date")
    fig.suptitle(f"Sentinel-1 RTC geolocation check ({mode_txt})", fontsize=11)
    fig.tight_layout()
    fname = f"test/s1_ts_stable_{'full' if full_timeseries else '2023_2024'}.png"
    fig.savefig(fname, dpi=150)


def spatial_stable():
    """
    Zoomed before/after maps (60 x 60 pixels) of Sentinel-1 RTC VV at stable
    reference structures in Switzerland, with per-site orbit track and dates.
    
    Per site you set:
      orbit  : track, e.g. ("ascending", 15), or None to pick automatically the
               first track with valid scenes on both sides
      before : "before" image = LAST valid acquisition on or before this date
      after  : "after" image  = FIRST valid acquisition on or after this date
    The reference time-series script prints the track and the two dates around
    the largest jump; copy them here.
    
    References drawn on the maps (all fixed in map coordinates, identical on both
    panels): a crosshair at the site centre and a 50 m grid. Optionally, the
    outline of the bright features in the BEFORE image is drawn on both panels,
    so any displacement of the after image relative to it is directly visible.
    You can also add your own reference lines (e.g. building edges from
    swisstopo swissTLM3D) per site.
    """
    
    # ---------------------------------------------------------------------------
    # Settings per site  (edit orbit / before / after here)
    # ---------------------------------------------------------------------------

    """
    Zurich airport terminal: 167 items
  largest jump: 1.1139 on track asc 15, between 2023-09-01 and 2023-09-13
Palexpo, Geneva: 227 items
  largest jump: 0.7180 on track asc 88, between 2024-04-21 and 2024-05-27
Verzasca (Contra) dam: 170 items
  largest jump: 0.0855 on track desc 66, between 2024-10-05 and 2024-10-17
Grande Dixence dam: 231 items
  largest jump: 0.0534 on track asc 88, between 2024-03-28 and 2024-04-09
St. Moritz station: 226 items
  largest jump: 0.0693 on track asc 15, between 2024-06-15 and 2024-07-21
    """
    
    SITES = {
        "Zurich airport terminal": {
            "center": (47.4506, 8.5617),
            "orbit": ("ascending", 15),
            "before": "2023-09-01",
            "after": "2023-09-13",
            "lines": {},  # optional: {"name": [(lat, lon), (lat, lon)]}
        },
        "Palexpo, Geneva": {
            "center": (46.2349, 6.1182),
            "orbit": ("ascending", 88),                 # TODO: set from the time-series script
            "before": "2024-04-21",        # TODO
            "after": "2024-05-27",         # TODO
            "lines": {},
        },
        "Verzasca (Contra) dam": {
            "center": (46.1967, 8.8478),
            "orbit": ("descending", 66),
            "before": "2024-10-05",
            "after": "2024-10-17",
            "lines": {},
        },
        "Grande Dixence dam": {
            "center": (46.0806, 7.4032),
            "orbit": ("ascending", 88),
            "before": "2024-03-28",
            "after": "2024-04-09",
            "lines": {},
        },
        "St. Moritz station": {
            "center": (46.4980, 9.8464),
            "orbit": ("ascending", 15),
            "before": "2024-06-15",
            "after": "2024-07-21",
            "lines": {},
        },
    }
    
    WINDOW_PX = 60                 # map size in pixels (each side)
    RESOLUTION = 10                # metres per pixel
    CRS = "EPSG:32632"
    SEARCH_DAYS = 90               # how far before/after the dates to look for scenes
    MIN_VALID_FRACTION = 0.95
    PROFILE_HALF_LENGTH_M = 250    # E-W and N-S profiles through the centre
    GRID_SPACING_M = 50
    SHOW_BEFORE_CONTOUR = True     # outline of bright features of the BEFORE image
    CONTOUR_PERCENTILE = 90
    
    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    to_utm = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    to_ll = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    HALF = WINDOW_PX * RESOLUTION / 2
    
    
    def window_bbox_ll(cx, cy, margin=100):
        xs = [cx - HALF - margin, cx + HALF + margin]
        ys = [cy - HALF - margin, cy + HALF + margin]
        lons, lats = to_ll.transform([xs[0], xs[1], xs[0], xs[1]], [ys[0], ys[0], ys[1], ys[1]])
        return [min(lons), min(lats), max(lons), max(lats)]
    
    
    def crop(img, cx, cy):
        y_desc = img.y.values[0] > img.y.values[-1]
        yslice = slice(cy + HALF, cy - HALF) if y_desc else slice(cy - HALF, cy + HALF)
        return img.sel(x=slice(cx - HALF, cx + HALF), y=yslice)
    
    
    def load_date(items, bbox, cx, cy):
        """Load all frames of one pass, crop to the window, return dB or None."""
        ds = odc.stac.load(items, bands=["vv"], bbox=bbox, crs=CRS,
                           resolution=RESOLUTION, groupby="solar_day")
        vv = crop(ds.vv.isel(time=0), cx, cy)
        vv = vv.where(vv > 0)
        if float(vv.notnull().mean()) < MIN_VALID_FRACTION:
            return None
        return 10 * np.log10(vv)
    
    
    def pick(by_date, dates, bbox, cx, cy):
        for d in dates:
            img = load_date(by_date[d], bbox, cx, cy)
            if img is not None:
                return img, d
        return None, None
    
    
    def find_pair(items, orbit, before_day, after_day, bbox, cx, cy):
        """Return (orbit, before_img, before_date, after_img, after_date) or None."""
        tracks = defaultdict(list)
        for it in items:
            p = it.properties
            tracks[(p.get("sat:orbit_state"), p.get("sat:relative_orbit"))].append(it)
        candidates = [orbit] if orbit is not None else sorted(tracks, key=lambda o: -len(tracks[o]))
        for orb in candidates:
            by_date = defaultdict(list)
            for it in tracks.get(orb, []):
                by_date[it.datetime.date()].append(it)
            b_img, b_date = pick(by_date, sorted([d for d in by_date if d <= before_day], reverse=True), bbox, cx, cy)
            if b_img is None:
                continue
            a_img, a_date = pick(by_date, sorted([d for d in by_date if d >= after_day]), bbox, cx, cy)
            if a_img is not None:
                return orb, b_img, b_date, a_img, a_date
        return None
    
    
    def estimate_shift(a, b):
        try:
            from skimage.registration import phase_cross_correlation
        except ImportError:
            return None
        av = np.nan_to_num(a.values, nan=np.nanmedian(a.values))
        bv = np.nan_to_num(b.values, nan=np.nanmedian(b.values))
        shift_px, _, _ = phase_cross_correlation(av, bv, upsample_factor=20)
        return (-shift_px[1] * float(a.x[1] - a.x[0]),
                -shift_px[0] * float(a.y[1] - a.y[0]))
    
    
    def profile(img, xs, ys):
        return img.interp(x=xr.DataArray(xs, dims="s"), y=xr.DataArray(ys, dims="s")).values
    
    
    # ---------------------------------------------------------------------------
    # Plot: one row per site  [before | after | E-W profile | N-S profile]
    # ---------------------------------------------------------------------------
    line_colors = ["red", "cyan", "yellow", "magenta"]
    fig, axes = plt.subplots(len(SITES), 4, figsize=(21, 5.2 * len(SITES)),
                             gridspec_kw={"width_ratios": [1, 1, 0.8, 0.8]}, squeeze=False)
    s = np.arange(-PROFILE_HALF_LENGTH_M, PROFILE_HALF_LENGTH_M + 1, RESOLUTION / 2)
    
    for row, (name, site) in zip(axes, SITES.items()):
        lat, lon = site["center"]
        cx, cy = to_utm.transform(lon, lat)
        bbox = window_bbox_ll(cx, cy)
        before_day = pd.Timestamp(site["before"]).date()
        after_day = pd.Timestamp(site["after"]).date()
        start = (pd.Timestamp(before_day) - pd.Timedelta(days=SEARCH_DAYS)).date()
        end = (pd.Timestamp(after_day) + pd.Timedelta(days=SEARCH_DAYS)).date()
    
        items = catalog.search(
            collections=["sentinel-1-rtc"], bbox=bbox, datetime=f"{start}/{end}",
            query={"sar:instrument_mode": {"eq": "IW"},
                   "sar:polarizations": {"eq": ["VV", "VH"]}},
        ).item_collection()
        result = find_pair(items, site["orbit"], before_day, after_day, bbox, cx, cy)
    
        if result is None:
            tracks = sorted({(it.properties.get("sat:orbit_state"), it.properties.get("sat:relative_orbit")) for it in items})
            msg = f"{name}: no valid before/after pair" + (f" on {site['orbit']}" if site["orbit"] else "")
            print(msg + f"\n  tracks available here: {tracks}")
            for ax in row:
                ax.axis("off")
            row[0].text(0.0, 0.5, msg, transform=row[0].transAxes, fontsize=10)
            continue
    
        orb, b_img, b_date, a_img, a_date = result
        orbit_txt = f"{orb[0]} {orb[1]}"
        print(f"{name}: {orbit_txt}, before {b_date}, after {a_date}")
    
        imgs = [b_img, a_img]
        labels = [f"Before: {b_date}", f"After: {a_date}"]
        vmin, vmax = np.nanpercentile(np.concatenate([i.values.ravel() for i in imgs]), [2, 98])
        contour_level = np.nanpercentile(b_img.values, CONTOUR_PERCENTILE)
        lines = {ln: np.array([to_utm.transform(lo, la) for la, lo in pts])
                 for ln, pts in site.get("lines", {}).items()}
    
        for ax, img, lab in zip(row[:2], imgs, labels):
            img.plot.imshow(ax=ax, cmap="gray", vmin=vmin, vmax=vmax,
                            add_colorbar=False, interpolation="nearest")
            # Fixed references in map coordinates
            grid = np.arange(-HALF, HALF + 1, GRID_SPACING_M)
            for g in grid:
                ax.axvline(cx + g, color="white", lw=0.3, alpha=0.4)
                ax.axhline(cy + g, color="white", lw=0.3, alpha=0.4)
            ax.plot([cx - PROFILE_HALF_LENGTH_M, cx + PROFILE_HALF_LENGTH_M], [cy, cy],
                    color="orange", lw=0.9, ls="--", label="E-W profile")
            ax.plot([cx, cx], [cy - PROFILE_HALF_LENGTH_M, cy + PROFILE_HALF_LENGTH_M],
                    color="deepskyblue", lw=0.9, ls="--", label="N-S profile")
            if SHOW_BEFORE_CONTOUR:
                ax.contour(b_img.x, b_img.y, b_img.values, levels=[contour_level],
                           colors="red", linewidths=0.8)
            for (ln, xy), c in zip(lines.items(), line_colors[1:]):
                ax.plot(xy[:, 0], xy[:, 1], color=c, lw=1, label=ln)
            ax.set_xlim(cx - HALF, cx + HALF)
            ax.set_ylim(cy - HALF, cy + HALF)
            ax.set_aspect("equal")
            ticks = np.arange(-HALF, HALF + 1, 100)
            ax.set_xticks(cx + ticks, [f"{t:+.0f}" for t in ticks])
            ax.set_yticks(cy + ticks, [f"{t:+.0f}" for t in ticks])
            ax.set_xlabel("m east of centre")
            ax.set_ylabel("m north of centre")
            ax.set_title(f"{name}\n{lab} ({orbit_txt})", fontsize=10)
        row[0].legend(fontsize=7, loc="lower left")
        if SHOW_BEFORE_CONTOUR:
            row[1].text(0.02, 0.98, "red: bright outline of BEFORE image", color="red",
                        fontsize=7, va="top", transform=row[1].transAxes,
                        bbox=dict(facecolor="black", alpha=0.5, lw=0))
    
        # Profiles: E-W (x varies) and N-S (y varies)
        for ax, (xs, ys), direction in [
            (row[2], (cx + s, np.full_like(s, cy)), "E-W (+ = east)"),
            (row[3], (np.full_like(s, cx), cy + s), "N-S (+ = north)"),
        ]:
            for img, lab in zip(imgs, labels):
                ax.plot(s, profile(img, xs, ys), marker=".", ms=3, label=lab)
            ax.axvline(0, color="gray", lw=0.6)
            ax.set_xlabel(f"Distance from centre, {direction} (m)")
            ax.set_ylabel("VV (dB)")
            ax.legend(fontsize=7)
    
        shift = estimate_shift(b_img, a_img)
        shift_txt = (f"after vs before: {shift[0]:+.1f} m E, {shift[1]:+.1f} m N"
                     if shift is not None else "install scikit-image for a shift estimate")
        print(f"  {shift_txt}")
        row[2].set_title(f"E-W profile\n{shift_txt}", fontsize=10)
        row[3].set_title("N-S profile", fontsize=10)
    
    fig.suptitle(f"Sentinel-1 RTC VV at reference structures, {WINDOW_PX} x {WINDOW_PX} pixel windows",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.7 / (5.2 * len(SITES))])  # room for title
    fig.savefig("test/s1_spatial_stable.png", dpi=130)


#-------------------------------------------------------------------------------------
# Check S1 tiles from Fola instead of Planetary Computer
#-------------------------------------------------------------------------------------


from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point
from pyproj import Transformer

GRID_SHP = '/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles_CH.shp'
ZARR_DIR = '/mnt/eo-nas1/data/satellite/sentinel1/raw/CH'

grid = gpd.read_file(GRID_SHP)


def find_tiles(lat, lon):

    site = (
        gpd.GeoDataFrame(
            geometry=[Point(lon, lat)],
            crs="EPSG:4326"
        )
        .to_crs(grid.crs)
    )

    hit = gpd.sjoin(
        site,
        grid,
        predicate="intersects"
    )

    if len(hit) == 0:
        raise ValueError(
            f"No grid tile found for ({lat}, {lon})"
        )

    idx = hit.index_right.iloc[0]
    row = grid.loc[idx]

    xmin, ymin, xmax, ymax = row.geometry.bounds

    x0 = int(xmin)
    y0 = int(ymax)

    matches = sorted(
        Path(ZARR_DIR).glob(
            f"S1_{x0}_{y0}_*.zarr"
        )
    )

    if not matches:
        raise FileNotFoundError(
            f"No zarr files found for {x0}_{y0}"
        )

    return matches

def open_timeseries(zarr_files, group):

    datasets = []

    for f in zarr_files:

        try:
            ds = xr.open_zarr(
                f,
                group=group
            )

            if ds.time.size > 0:
                datasets.append(ds)

        except Exception as e:
            print(f"Failed: {f.name} ({group}) : {e}")

    if len(datasets) == 0:
        return None

    ds = xr.concat(
        datasets,
        dim="time"
    )

    ds = ds.sortby("time")

    return ds


def compute_deltas_local(ds, bbox):

    xmin, ymin, xmax, ymax = bbox

    lat_mask = (ds.lat >= ymin) & (ds.lat <= ymax)
    lon_mask = (ds.lon >= xmin) & (ds.lon <= xmax)
    
    if lat_mask.sum() == 0 or lon_mask.sum() == 0:
        return None

    vv = ds.s1_vv.sel(
        lat=ds.lat[lat_mask],
        lon=ds.lon[lon_mask]
    )

    vv = vv.where(
        ds.s1_valid.sel(
            lat=ds.lat[lat_mask],
            lon=ds.lon[lon_mask]
        ) > 0
    )

    mean_vv = vv.mean(
        dim=("lat", "lon"),
        skipna=True
    )

    if mean_vv.time.size < N_BASELINE + 2:
        return None

    baseline = (
        mean_vv
        .isel(time=slice(0, N_BASELINE))
        .mean()
    )

    return np.abs(mean_vv - baseline)


def plot_shift_timeseries_loc(
    full_timeseries=False,
    sites=SITES,
):
    """
    Plot Sentinel-1 VV stability time series from local Zarr archive.

    full_timeseries=True:
        Full available record, thinned using SAMPLE_EVERY/SAMPLE_UNIT

    full_timeseries=False:
        Only SHORT_PERIOD (e.g. 2023-2024), all acquisitions

    Curves are separated by:
        ASC/DESC + relative orbit
    """

    period = FULL_PERIOD if full_timeseries else SHORT_PERIOD

    mode_txt = (
        f"{period}, every {SAMPLE_EVERY} {SAMPLE_UNIT}"
        if full_timeseries
        else f"{period}, all acquisitions"
    )

    fig, axes = plt.subplots(
        len(sites),
        1,
        figsize=(12, 3.2 * len(sites)),
        sharex=True,
    )

    axes = np.atleast_1d(axes)

    for ax, (site, cfg) in zip(axes, sites.items()):

        lat, lon = cfg["center"]

        bbox = make_bbox_utm(
            lat,
            lon,
            cfg.get("half_km", 0.4),
        )

        print(f"\n{site}")

        zarr_files = find_tiles(lat, lon)

        print(f"  found {len(zarr_files)} yearly tiles")

        groups = {}

        asc_ds = open_timeseries(
            zarr_files,
            "asc",
        )

        if asc_ds is not None:
            groups["ASC"] = asc_ds

        desc_ds = open_timeseries(
            zarr_files,
            "desc",
        )

        if desc_ds is not None:
            groups["DESC"] = desc_ds

        best = None

        for direction, ds in groups.items():

            orbits = np.unique(
                ds.relative_orbit.values
            )

            print(
                f"  {direction}: "
                f"{len(orbits)} relative orbits"
            )

            for orbit in sorted(orbits):

                orbit_mask = (
                    ds.relative_orbit.values == orbit
                )

                orbit_ds = ds.isel(
                    time=orbit_mask
                )

                if orbit_ds.time.size < N_BASELINE + 2:
                    continue

                deltas = compute_deltas_local(
                    orbit_ds,
                    bbox,
                )

                if deltas is None:
                    continue

                # Restrict to SHORT_PERIOD
                if not full_timeseries:

                    start, end = SHORT_PERIOD.split("/")

                    deltas = deltas.sel(
                        time=slice(
                            np.datetime64(start),
                            np.datetime64(end),
                        )
                    )

                    if deltas.time.size < N_BASELINE + 2:
                        continue

                # Thin only for full record
                if full_timeseries:

                    keep = thin_dates(
                        sorted(
                            pd.to_datetime(
                                deltas.time.values
                            )
                        ),
                        SAMPLE_EVERY,
                        SAMPLE_UNIT,
                    )

                    deltas = deltas.sel(
                        time=keep
                    )

                if deltas.time.size == 0:
                    continue

                ax.plot(
                    deltas.time,
                    deltas.values,
                    ".",
                    label=(
                        f"{direction}-{orbit} "
                        f"(n={deltas.sizes['time']})"
                    ),
                )

                jump = largest_jump(deltas)

                if (
                    jump is not None
                    and (
                        best is None
                        or jump[0] > best[0]
                    )
                ):
                    best = (
                        *jump,
                        f"{direction}-{orbit}",
                    )

        if best is not None:

            size, t_before, t_after, track = best

            date_txt = pd.Timestamp(
                t_after
            ).strftime("%Y-%m-%d")

            ax.axvline(
                t_after,
                color="red",
                lw=1,
                ls="--",
            )

            ax.text(
                t_after,
                0.97,
                date_txt,
                rotation=90,
                color="red",
                fontsize=9,
                ha="right",
                va="top",
                transform=ax.get_xaxis_transform(),
            )

            print(
                f"  largest jump: {size:.4f} "
                f"on {track}, between "
                f"{pd.Timestamp(t_before):%Y-%m-%d} "
                f"and {date_txt}"
            )

            ax.legend(
                fontsize=7,
                loc="upper left",
                ncol=3,
            )

        else:

            ax.text(
                0.5,
                0.5,
                "no usable data",
                transform=ax.transAxes,
                ha="center",
            )

        ax.set_title(
            f"{site}: mean |VV - baseline "
            f"of first {N_BASELINE} images|",
            fontsize=10,
        )

        ax.set_ylabel(
            "Mean |ΔVV| (linear)"
        )

    axes[-1].set_xlabel(
        "Acquisition date"
    )

    fig.suptitle(
        f"Sentinel-1 RTC geolocation check ({mode_txt})",
        fontsize=11,
    )

    fig.tight_layout()

    fname = f"test/fola/s1_ts_stable_{'full' if full_timeseries else '2023_2024'}.png"
    fig.savefig(fname, dpi=150)


def spatial_stable_fola():
    """
    Zoomed before/after maps (60 x 60 pixels) of Sentinel-1 RTC VV at stable
    reference structures in Switzerland, with per-site orbit track and dates.
    
    Per site you set:
      orbit  : track, e.g. ("ascending", 15), or None to pick automatically the
               first track with valid scenes on both sides
      before : "before" image = LAST valid acquisition on or before this date
      after  : "after" image  = FIRST valid acquisition on or after this date
    The reference time-series script prints the track and the two dates around
    the largest jump; copy them here.
    
    References drawn on the maps (all fixed in map coordinates, identical on both
    panels): a crosshair at the site centre and a 50 m grid. Optionally, the
    outline of the bright features in the BEFORE image is drawn on both panels,
    so any displacement of the after image relative to it is directly visible.
    You can also add your own reference lines (e.g. building edges from
    swisstopo swissTLM3D) per site.
    """
    
    # ---------------------------------------------------------------------------
    # Settings per site  (edit orbit / before / after here)
    # ---------------------------------------------------------------------------
    SITES = {
        "Zurich airport terminal": {
            "center": (47.45033, 8.56131),
            "orbit": ("ascending", 88),
            "before": "2019-09-03",
            "after": "2019-09-27",
            "lines": {},  # optional: {"name": [(lat, lon), (lat, lon)]}
        },
        "Palexpo, Geneva": {
            "center": (46.2349, 6.1182),
            "orbit": ("ascending", 161),                 # TODO: set from the time-series script
            "before": "2025-01-03",        # TODO
            "after": "2025-01-27",         # TODO
            "lines": {},
        },
        "Verzasca (Contra) dam": {
            "center": (46.1967, 8.8478),
            "orbit": ("descending", 66),
            "before": "2019-02-22",
            "after": "2019-03-18",
            "lines": {},
        },
    }
    
    WINDOW_PX = 60                 # map size in pixels (each side)
    RESOLUTION = 10                # metres per pixel
    CRS = "EPSG:32632"
    SEARCH_DAYS = 90               # how far before/after the dates to look for scenes
    MIN_VALID_FRACTION = 0.95
    PROFILE_HALF_LENGTH_M = 250    # E-W and N-S profiles through the centre
    GRID_SPACING_M = 50
    SHOW_BEFORE_CONTOUR = True     # outline of bright features of the BEFORE image
    CONTOUR_PERCENTILE = 90
    
    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    to_utm = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    to_ll = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    HALF = WINDOW_PX * RESOLUTION / 2
    
    
    def window_bbox_ll(cx, cy, margin=100):
        xs = [cx - HALF - margin, cx + HALF + margin]
        ys = [cy - HALF - margin, cy + HALF + margin]
        lons, lats = to_ll.transform([xs[0], xs[1], xs[0], xs[1]], [ys[0], ys[0], ys[1], ys[1]])
        return [min(lons), min(lats), max(lons), max(lats)]
        
    
    def load_date(items, bbox, cx, cy):
        """Load all frames of one pass, crop to the window, return dB or None."""
        ds = odc.stac.load(items, bands=["vv"], bbox=bbox, crs=CRS,
                           resolution=RESOLUTION, groupby="solar_day")
        vv = crop(ds.vv.isel(time=0), cx, cy)
        vv = vv.where(vv > 0)
        if float(vv.notnull().mean()) < MIN_VALID_FRACTION:
            return None
        return 10 * np.log10(vv)
    
    
    def pick(by_date, dates, bbox, cx, cy):
        for d in dates:
            img = load_date(by_date[d], bbox, cx, cy)
            if img is not None:
                return img, d
        return None, None

    def crop(img, cx, cy):

        yslice = slice(
            cy + HALF,
            cy - HALF,
        )
    
        return img.sel(
            lon=slice(
                cx - HALF,
                cx + HALF,
            ),
            lat=yslice,
        )
    
    def find_pair_local(ds, orbit, before_day, after_day, cx, cy):
    
        direction, rel_orbit = orbit
    
        group_name = (
            "ASC"
            if direction.lower().startswith("asc")
            else "DESC"
        )
    
        orbit_mask = ds.relative_orbit.values == rel_orbit
    
        orbit_ds = ds.isel(time=orbit_mask)
    
        if orbit_ds.time.size == 0:
            return None
    
        times = pd.to_datetime(
            orbit_ds.time.values
        )
    
        before_idx = np.where(
            times.date <= before_day
        )[0]
    
        if len(before_idx) == 0:
            return None
    
        before_idx = before_idx[-1]
    
        after_idx = np.where(
            times.date >= after_day
        )[0]
    
        if len(after_idx) == 0:
            return None
    
        after_idx = after_idx[0]
    
        before_img = crop(
            orbit_ds.s1_vv.isel(time=before_idx),
            cx,
            cy,
        )
        
        after_img = crop(
            orbit_ds.s1_vv.isel(time=after_idx),
            cx,
            cy,
        )
        
        before_valid = crop(
            orbit_ds.s1_valid.isel(time=before_idx),
            cx,
            cy,
        )
        
        after_valid = crop(
            orbit_ds.s1_valid.isel(time=after_idx),
            cx,
            cy,
        )
    
        before_img = before_img.where(
            before_valid > 0
        )
    
        after_img = after_img.where(
            after_valid > 0
        )
    
        before_img = 10 * np.log10(
            before_img.where(before_img > 0)
        )
    
        after_img = 10 * np.log10(
            after_img.where(after_img > 0)
        )
    
        return (
            orbit,
            before_img,
            times[before_idx].date(),
            after_img,
            times[after_idx].date(),
        )
    
    def load_site_data(lat, lon):
    
        zarr_files = find_tiles(
            lat,
            lon,
        )
    
        return {
            "ASC": open_timeseries(
                zarr_files,
                "asc",
            ),
            "DESC": open_timeseries(
                zarr_files,
                "desc",
            ),
        }

    
    def estimate_shift(a, b):
        try:
            from skimage.registration import phase_cross_correlation
        except ImportError:
            return None
        av = np.nan_to_num(a.values, nan=np.nanmedian(a.values))
        bv = np.nan_to_num(b.values, nan=np.nanmedian(b.values))
        shift_px, _, _ = phase_cross_correlation(av, bv, upsample_factor=20)
        return (-shift_px[1] * float(a.lon[1] - a.lon[0]),
                -shift_px[0] * float(a.lat[1] - a.lat[0]))
    
    
    def profile(img, xs, ys):
        return img.interp(lon=xr.DataArray(xs, dims="s"), lat=xr.DataArray(ys, dims="s")).values
    
    
    # ---------------------------------------------------------------------------
    # Plot: one row per site  [before | after | E-W profile | N-S profile]
    # ---------------------------------------------------------------------------
    line_colors = ["red", "cyan", "yellow", "magenta"]
    fig, axes = plt.subplots(len(SITES), 4, figsize=(21, 5.2 * len(SITES)),
                             gridspec_kw={"width_ratios": [1, 1, 0.8, 0.8]}, squeeze=False)
    s = np.arange(-PROFILE_HALF_LENGTH_M, PROFILE_HALF_LENGTH_M + 1, RESOLUTION / 2)
    
    for row, (name, site) in zip(axes, SITES.items()):
        lat, lon = site["center"]
        cx, cy = to_utm.transform(lon, lat)
        bbox = window_bbox_ll(cx, cy)
        before_day = pd.Timestamp(site["before"]).date()
        after_day = pd.Timestamp(site["after"]).date()
        start = (pd.Timestamp(before_day) - pd.Timedelta(days=SEARCH_DAYS)).date()
        end = (pd.Timestamp(after_day) + pd.Timedelta(days=SEARCH_DAYS)).date()
    
        groups = load_site_data(
            lat,
            lon,
        )
        
        direction = (
            "ASC"
            if site["orbit"][0].lower().startswith("asc")
            else "DESC"
        )
        
        ds = groups.get(direction)
        
        if ds is None:
        
            result = None
        
        else:
        
            result = find_pair_local(
                ds,
                site["orbit"],
                before_day,
                after_day,
                cx,
                cy,
            )
    
        if result is None:
            tracks = sorted({(it.properties.get("sat:orbit_state"), it.properties.get("sat:relative_orbit")) for it in items})
            msg = f"{name}: no valid before/after pair" + (f" on {site['orbit']}" if site["orbit"] else "")
            print(msg + f"\n  tracks available here: {tracks}")
            for ax in row:
                ax.axis("off")
            row[0].text(0.0, 0.5, msg, transform=row[0].transAxes, fontsize=10)
            continue
    
        orb, b_img, b_date, a_img, a_date = result
        orbit_txt = f"{orb[0]} {orb[1]}"
        print(f"{name}: {orbit_txt}, before {b_date}, after {a_date}")
    
        imgs = [b_img, a_img]
        labels = [f"Before: {b_date}", f"After: {a_date}"]
        vmin, vmax = np.nanpercentile(np.concatenate([i.values.ravel() for i in imgs]), [2, 98])
        contour_level = np.nanpercentile(b_img.values, CONTOUR_PERCENTILE)
        lines = {ln: np.array([to_utm.transform(lo, la) for la, lo in pts])
                 for ln, pts in site.get("lines", {}).items()}
    
        for ax, img, lab in zip(row[:2], imgs, labels):
            img.plot.imshow(ax=ax, cmap="gray", vmin=vmin, vmax=vmax,
                            add_colorbar=False, interpolation="nearest")
            # Fixed references in map coordinates
            grid = np.arange(-HALF, HALF + 1, GRID_SPACING_M)
            for g in grid:
                ax.axvline(cx + g, color="white", lw=0.3, alpha=0.4)
                ax.axhline(cy + g, color="white", lw=0.3, alpha=0.4)
            ax.plot([cx - PROFILE_HALF_LENGTH_M, cx + PROFILE_HALF_LENGTH_M], [cy, cy],
                    color="orange", lw=0.9, ls="--", label="E-W profile")
            ax.plot([cx, cx], [cy - PROFILE_HALF_LENGTH_M, cy + PROFILE_HALF_LENGTH_M],
                    color="deepskyblue", lw=0.9, ls="--", label="N-S profile")
            if SHOW_BEFORE_CONTOUR:
                ax.contour(b_img.lon, b_img.lat, b_img.values, levels=[contour_level],
                           colors="red", linewidths=0.8)
            for (ln, xy), c in zip(lines.items(), line_colors[1:]):
                ax.plot(xy[:, 0], xy[:, 1], color=c, lw=1, label=ln)
            ax.set_xlim(cx - HALF, cx + HALF)
            ax.set_ylim(cy - HALF, cy + HALF)
            ax.set_aspect("equal")
            ticks = np.arange(-HALF, HALF + 1, 100)
            ax.set_xticks(cx + ticks, [f"{t:+.0f}" for t in ticks])
            ax.set_yticks(cy + ticks, [f"{t:+.0f}" for t in ticks])
            ax.set_xlabel("m east of centre")
            ax.set_ylabel("m north of centre")
            ax.set_title(f"{name}\n{lab} ({orbit_txt})", fontsize=10)
        row[0].legend(fontsize=7, loc="lower left")
        if SHOW_BEFORE_CONTOUR:
            row[1].text(0.02, 0.98, "red: bright outline of BEFORE image", color="red",
                        fontsize=7, va="top", transform=row[1].transAxes,
                        bbox=dict(facecolor="black", alpha=0.5, lw=0))
    
        # Profiles: E-W (x varies) and N-S (y varies)
        for ax, (xs, ys), direction in [
            (row[2], (cx + s, np.full_like(s, cy)), "E-W (+ = east)"),
            (row[3], (np.full_like(s, cx), cy + s), "N-S (+ = north)"),
        ]:
            for img, lab in zip(imgs, labels):
                ax.plot(s, profile(img, xs, ys), marker=".", ms=3, label=lab)
            ax.axvline(0, color="gray", lw=0.6)
            ax.set_xlabel(f"Distance from centre, {direction} (m)")
            ax.set_ylabel("VV (dB)")
            ax.legend(fontsize=7)
    
        shift = estimate_shift(b_img, a_img)
        shift_txt = (f"after vs before: {shift[0]:+.1f} m E, {shift[1]:+.1f} m N"
                     if shift is not None else "install scikit-image for a shift estimate")
        print(f"  {shift_txt}")
        row[2].set_title(f"E-W profile\n{shift_txt}", fontsize=10)
        row[3].set_title("N-S profile", fontsize=10)
    
    fig.suptitle(f"Sentinel-1 RTC VV at reference structures, {WINDOW_PX} x {WINDOW_PX} pixel windows",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.7 / (5.2 * len(SITES))])  # room for title
    fig.savefig("test/fola/s1_spatial_stable.png", dpi=130)

def analyse_year(year):
    # centre point (WGS84)
    lat0 = 47.45033
    lon0 = 8.56131
    region = "Zurich Airport Terminal"

    # lat0 = 46.23503
    # lon0 = 6.11827
    # region = "Geneva Palexpo"

    x0, y0 = utm_transformer.transform(lon0, lat0)
    WINDOW_M = 50

    zarr_files = find_tiles(lat0, lon0)
    ds = open_timeseries(zarr_files, "asc")

    # orbit 88 over all years (needed for the S1A reference date)
    orbit_mask = ds.relative_orbit.compute().values == 88
    times = pd.to_datetime(ds.time.values)
    ds = ds.isel(time=orbit_mask & (times.year == year))

    n = ds.time.size
    print(f"{n} ASC-88 acquisitions in {year}")
    if n == 0:
        return

    platform = np.asarray(ds.platform.compute().values).astype(str)
    platform = np.char.upper(np.char.strip(platform))
    print(np.unique(platform))  # check how the values are spelled
    
    # works for "S1A", "Sentinel-1A", "SENTINEL-1A", etc.
    is_a = np.char.endswith(platform, "A")
    is_b = np.char.endswith(platform, "B")
    
    idx_a = np.where(is_a)[0]
    idx_b = np.where(is_b)[0]
    print(f"  S1A: {len(idx_a)}   S1B: {len(idx_b)}")

    ncols_per_sat = 4
    ncols = 2 * ncols_per_sat
    nrows = max(
        math.ceil(len(idx_a) / ncols_per_sat),
        math.ceil(len(idx_b) / ncols_per_sat),
        1,
    )

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3 * ncols, 3 * nrows),
        squeeze=False,
    )

    # switch everything off first; panels used get switched back on
    for ax in axes.ravel():
        ax.axis("off")

    def plot_panel(ax, i, sat):
        img = ds.s1_vv.isel(time=i).where(ds.s1_valid.isel(time=i) > 0)
        img = img.sel(
            lon=slice(x0 - WINDOW_M, x0 + WINDOW_M),
            lat=slice(y0 + WINDOW_M, y0 - WINDOW_M),
        )
        img = 10 * np.log10(img.where(img > 0))

        ax.axis("on")
        ax.imshow(img.values, cmap="gray", origin="upper")

        ix = np.argmin(np.abs(img.lon.values - x0))
        iy = np.argmin(np.abs(img.lat.values - y0))
        ax.axvline(ix, color="red", lw=1)
        ax.axhline(iy, color="red", lw=1)

        ax.set_title(
            f"{sat} {pd.Timestamp(ds.time.values[i]).strftime('%Y-%m-%d')}",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # S1A -> columns 0-3
    for k, i in enumerate(idx_a):
        r, c = divmod(k, ncols_per_sat)
        plot_panel(axes[r, c], i, "S1A")

    # S1B -> columns 4-7
    for k, i in enumerate(idx_b):
        r, c = divmod(k, ncols_per_sat)
        plot_panel(axes[r, c + ncols_per_sat], i, "S1B")

    plt.suptitle(
        f"Sentinel-1 ASC-88 ({year})\n{region}",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # column-group headers and a separator line between S1A and S1B
    fig.text(0.25, 0.945, f"S1A (n={len(idx_a)})",
             ha="center", fontsize=12, weight="bold")
    fig.text(0.75, 0.945, f"S1B (n={len(idx_b)})",
             ha="center", fontsize=12, weight="bold")
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.02, 0.94],
                              color="black", lw=1.5,
                              transform=fig.transFigure))

    plt.savefig(
        f"test/fola/yearly/s1ab_asc88_{region[:3]}_{year}_zoom_2_S1AB.png",
        dpi=150,
        bbox_inches="tight",
    )

def plot_building_centroid_timeseries(
    ds,
    region,
    orbit=88,
    year=2019,
    lat0=47.45033,
    lon0=8.56131,
    window_m=300,
    percentile=99.5,
):
    """
    Track centroid of bright building through time.

    Parameters
    ----------
    ds : xarray.Dataset
        ASC or DESC dataset loaded from Zarr.

    orbit : int
        Relative orbit.

    year : int

    lat0, lon0 : float
        Reference coordinate inside building.

    window_m : float
        Crop size around target.

    percentile : float
        Bright pixel threshold.
    """

    x0, y0 = utm_transformer.transform(
        lon0,
        lat0,
    )

    orbit_mask = (
        ds.relative_orbit.compute().values == orbit
    )

    ds = ds.isel(time=orbit_mask)

    times = pd.to_datetime(ds.time.values)

    ds = ds.isel(
        time=(times.year == year)
    )

    if ds.time.size == 0:
        print("No observations found.")
        return

    east = []
    north = []
    dates = []

    for i in range(ds.time.size):

        img = (
            ds.s1_vv
            .isel(time=i)
            .where(
                ds.s1_valid.isel(time=i) > 0
            )
        )

        img = img.sel(
            lon=slice(
                x0 - window_m,
                x0 + window_m,
            ),
            lat=slice(
                y0 + window_m,
                y0 - window_m,
            ),
        )

        arr = img.values

        if np.all(np.isnan(arr)):
            continue

        thresh = np.nanpercentile(
            arr,
            percentile,
        )

        mask = arr > thresh

        iy0 = np.argmin(
            np.abs(img.lat.values - y0)
        )

        ix0 = np.argmin(
            np.abs(img.lon.values - x0)
        )

        labels, nobj = label(mask)

        building_label = labels[iy0, ix0]

        if building_label == 0:
            continue

        building = (
            labels == building_label
        )

        cy, cx = center_of_mass(
            building
        )

        x_centroid = np.interp(
            cx,
            np.arange(len(img.lon)),
            img.lon.values,
        )

        y_centroid = np.interp(
            cy,
            np.arange(len(img.lat)),
            img.lat.values,
        )

        east.append(
            x_centroid - x0
        )

        north.append(
            y_centroid - y0
        )

        dates.append(
            pd.Timestamp(
                ds.time.values[i]
            )
        )

    east = np.array(east)
    north = np.array(north)

    print(
        f"East std:  {np.std(east):.2f} m"
    )
    print(
        f"North std: {np.std(north):.2f} m"
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 6),
        sharex=True,
    )

    axes[0].plot(
        dates,
        east,
        "o-",
    )

    axes[0].axhline(
        0,
        color="k",
        lw=0.8,
    )

    axes[0].set_ylabel(
        "East displacement (m)"
    )

    axes[0].set_title(
        f"Orbit {orbit}, {year}"
    )

    axes[1].plot(
        dates,
        north,
        "o-",
    )

    axes[1].axhline(
        0,
        color="k",
        lw=0.8,
    )

    axes[1].set_ylabel(
        "North displacement (m)"
    )

    axes[1].set_xlabel(
        "Date"
    )

    plt.tight_layout()
    plt.savefig(
        f"test/fola/yearly/brighest_shift_{region[:3]}_{year}.png",
        dpi=150,
        bbox_inches="tight",
    )

    return pd.DataFrame(
        {
            "date": dates,
            "east_m": east,
            "north_m": north,
        }
    )

def run_plot_centroid():
    # centre point (WGS84)
    lat0 = 47.45033
    lon0 = 8.56131
    region = "Zurich Airport Terminal"

    # lat0 = 46.23503
    # lon0 = 6.11827
    # region = "Geneva Palexpo"
    
    zarr_files = find_tiles(
        lat0,
        lon0,
    )
    
    ds = open_timeseries(
        zarr_files,
        "asc",
    )
    
    df = plot_building_centroid_timeseries(
        ds,
        region=region,
        orbit=88,
        year=2020,
        lat0=lat0,
        lon0=lon0,
        percentile=95,
    )

def plot_pixel_timeseries(
    lat0=47.45033,
    lon0=8.56131,
    orbit=88,
):

    # point in UTM
    x0, y0 = utm_transformer.transform(
        lon0,
        lat0,
    )

    zarr_files = find_tiles(
        lat0,
        lon0,
    )

    ds = open_timeseries(
        zarr_files,
        "asc",
    )

    # orbit filter
    orbit_mask = (
        ds.relative_orbit.compute().values
        == orbit
    )
    print(ds.platform.values)
    ds = ds.isel(
        time=orbit_mask
    )

        # orbit filter
    plat_mask = (
        ds.platform.compute().values
        == 'SENTINEL-1B'
    )

    ds = ds.isel(
        time=plat_mask
    )

    # nearest pixel
    ix = np.argmin(
        np.abs(ds.lon.values - x0)
    )

    iy = np.argmin(
        np.abs(ds.lat.values - y0)
    )

    vv = (
        ds.s1_vv[:, iy, ix]
        .where(
            ds.s1_valid[:, iy, ix] > 0
        )
        .compute()
    )

    # linear power -> dB
    vv_db = 10 * np.log10(
        vv.where(vv > 0)
    )

    plt.figure(figsize=(12, 4))

    plt.plot(
        ds.time.values,
        vv_db.values,
        ".-",
    )

    plt.grid(True)

    plt.xlabel("Date")
    plt.ylabel("VV (dB)")

    plt.title(
        f"Sentinel-1 ASC-{orbit}\n"
        f"nearest pixel to "
        f"({lat0:.5f}, {lon0:.5f})"
    )

    plt.tight_layout()
    plt.savefig("test/test_b.png", dpi=150)


def plot_swissimage(
    path="/mnt/eo-nas1/data/swisstopo/SwissImage/cubes/2m/SwissImage2_466140_5255340.zarr",
    lat0=47.45033,
    lon0=8.56131,
    region="Zurich Airport Terminal",
    WINDOW_M=100,
):
    path = '/mnt/eo-nas1/data/swisstopo/SwissImage/cubes/10cm/SwissImage0.1_466140_5255340.zarr'
    x0, y0 = utm_transformer.transform(lon0, lat0)

    ds = xr.open_zarr(path)
    print(ds)  # check variable / coordinate names on first run

    # --- get RGB as a DataArray with dims (y, x, band) ---
    rgb_names = None
    for cand in (("red", "green", "blue"), ("R", "G", "B"), ("r", "g", "b")):
        if all(c in ds.data_vars for c in cand):
            rgb_names = cand
            break

    if rgb_names is not None:
        rgb = xr.concat([ds[c] for c in rgb_names], dim="band")
    else:
        # single variable with a band dimension
        var = list(ds.data_vars)[0]
        rgb = ds[var]
        band_dim = next(d for d in rgb.dims if d in ("band", "bands", "channel"))
        rgb = rgb.rename({band_dim: "band"}).isel(band=slice(0, 3))

    # --- coordinate names (x/y or lon/lat) ---
    xdim = "x" if "x" in rgb.dims else "lon"
    ydim = "y" if "y" in rgb.dims else "lat"

    # --- crop window, respecting coordinate order ---
    xs = rgb[xdim].values
    ys = rgb[ydim].values
    xslice = slice(x0 - WINDOW_M, x0 + WINDOW_M) if xs[0] < xs[-1] \
        else slice(x0 + WINDOW_M, x0 - WINDOW_M)
    yslice = slice(y0 + WINDOW_M, y0 - WINDOW_M) if ys[0] > ys[-1] \
        else slice(y0 - WINDOW_M, y0 + WINDOW_M)

    rgb = rgb.sel({xdim: xslice, ydim: yslice})

    # make north up
    if rgb[ydim].values[0] < rgb[ydim].values[-1]:
        rgb = rgb.isel({ydim: slice(None, None, -1)})

    rgb = rgb.transpose(ydim, xdim, "band")
    v = rgb.compute().values.astype("float32")

    # --- scale to 0-1 for imshow ---
    if np.nanmax(v) > 1:
        if rgb.dtype == np.uint8 or np.nanmax(v) <= 255:
            v = v / 255.0
        else:  # e.g. uint16: simple 2-98 % stretch
            lo, hi = np.nanpercentile(v, (2, 98))
            v = (v - lo) / (hi - lo)
    v = np.clip(v, 0, 1)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(v, origin="upper")

    ix = np.argmin(np.abs(rgb[xdim].values - x0))
    iy = np.argmin(np.abs(rgb[ydim].values - y0))
    ax.axvline(ix, color="red", lw=1)
    ax.axhline(iy, color="red", lw=1)

    ax.set_title(f"SWISSIMAGE true colour\n{region}", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(
        f"test/fola/yearly/swissimage_{region[:3]}_zoom.png",
        dpi=150,
        bbox_inches="tight",
    )


def new_test_s1():
    import numpy as np
    import pandas as pd
    import xarray as xr
    import matplotlib.pyplot as plt
    import pystac_client, planetary_computer, odc.stac
    from odc.geo.geobox import GeoBox
    from pyproj import Transformer
    
    TILE_PATH = '/mnt/eo-nas1/data/satellite/sentinel1/raw/CH/S1_467420_5259180_20230104_20231230.zarr'    # adjust directory
    DATE = "2023-10-19"
    RELATIVE_ORBIT = 15
    CRS = "EPSG:32632"
    RES = 10
    
    # --- local tile, one date ---
    z = xr.open_zarr(TILE_PATH, group="asc")
    t = pd.to_datetime(z.time.values).normalize()
    sel = (t == pd.Timestamp(DATE)) & (z.relative_orbit.compute().values == RELATIVE_ORBIT)
    print(f"{sel.sum()} local time step(s) on {DATE}")
    
    loc = z.s1_vv.isel(time=sel).where(z.s1_valid.isel(time=sel) > 0)
    loc = loc.where(loc > 0).mean("time").rename(lon="x", lat="y")
    loc = loc.sortby("x").sortby("y", ascending=False).load()   # north-up
    print("local x:", float(loc.x.min()), "-", float(loc.x.max()),
          " y:", float(loc.y.min()), "-", float(loc.y.max()), " size:", dict(loc.sizes))
    
    # --- PC on the same grid (local coords treated as pixel centres) ---
    x0, x1 = float(loc.x.min()) - RES / 2, float(loc.x.max()) + RES / 2
    y0, y1 = float(loc.y.min()) - RES / 2, float(loc.y.max()) + RES / 2
    
    to_ll = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    lons, lats = to_ll.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
    bbox_ll = [min(lons), min(lats), max(lons), max(lats)]
    
    items = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    ).search(
        collections=["sentinel-1-rtc"],
        bbox=bbox_ll,
        datetime=f"{DATE}/{DATE}",
        query={"sat:orbit_state": {"eq": "ascending"},
               "sat:relative_orbit": {"eq": RELATIVE_ORBIT}},
    ).item_collection()
    print(f"{len(items)} PC item(s):", [i.id for i in items])
    
    gbox = GeoBox.from_bbox((x0, y0, x1, y1), crs=CRS, resolution=RES, tight=True)
    pc = odc.stac.load(list(items), bands=["vv"], geobox=gbox,
                       groupby="solar_day").vv.isel(time=0)
    pc = pc.where(pc > 0).load()
    
    # --- plot ---
    extent = [x0, x1, y0, y1]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for a, da, title in [(ax[0], pc, f"PC {DATE}"),
                         (ax[1], loc, f"Local tile 467420_5259180, {DATE}")]:
        im = a.imshow(10 * np.log10(da.values), cmap="gray", vmin=-25, vmax=0,
                      extent=extent)
        a.set_title(title, fontsize=10)
        a.set_aspect("equal")
        a.ticklabel_format(style="plain", useOffset=False)
        a.tick_params(labelsize=7)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="VV (dB)")
    plt.show()
    
    # --- quick agreement check ---
    a, b = 10 * np.log10(pc.values), 10 * np.log10(loc.values)
    m = np.isfinite(a) & np.isfinite(b)
    print(f"corr = {np.corrcoef(a[m], b[m])[0, 1]:.3f}, "
          f"median(local - PC) = {np.median(b[m] - a[m]):+.2f} dB, n = {m.sum()}")
    plt.savefig('test/s1_diagnosis2.png', dpi=150)

def full_diag():
    """
    Diagnostic: local Sentinel-1 zarr tiles vs. Planetary Computer sentinel-1-rtc.
    
    Checks, for one date and one orbit track over a bbox:
      1. Which 1280 m tiles are needed, which were found, which are missing
      2. Per-tile metadata: size, resolution, coordinate order, offset of the
         coordinates vs. the tile name, CRS metadata (relevant for QGIS)
      3. A local mosaic on a regular 10 m grid (gaps stay NaN, nothing is stretched)
      4. PC data loaded on exactly the same pixel grid
      5. Correlation and sub-pixel shift (phase correlation), overall and per tile
      6. Plots: tile layout, PC vs local vs difference
      7. Optional GeoTIFF export of both, with CRS and north-up rows, for QGIS
    
    Requires find_tiles(lat, lon) from your own code to be importable / defined.
    """
    
    import os
    import re
    
    import matplotlib.pyplot as plt
    import numpy as np
    import odc.stac
    import pandas as pd
    import planetary_computer
    import pystac_client
    import xarray as xr
    from matplotlib.patches import Rectangle
    from odc.geo.geobox import GeoBox
    from pyproj import Transformer
    
    # from your_module import find_tiles   # <- adjust to wherever find_tiles lives
    
    # ---------------------------------------------------------------------------
    # Settings
    # ---------------------------------------------------------------------------
    DATE = "2023-10-19"
    GROUP = "asc"                    # zarr group in the local files
    ORBIT_STATE_PC = "ascending"     # same pass direction on PC
    RELATIVE_ORBIT = 15
    
    BBOX = [8.525, 47.443, 8.580, 47.482]   # lon/lat: minx, miny, maxx, maxy
    CRS = "EPSG:32632"
    TILE_M = 1280                    # 128 px x 10 m
    RES = 10
    SAMPLE_STEP_M = 500              # must be < TILE_M so no tile is skipped
    OUT_DIR = "test/diag"
    
    REF_LINES = {
        "Runway 10/28": [(47.45900, 8.53750), (47.45667, 8.57050)],
        "Runway 16/34": [(47.47567, 8.53600), (47.44933, 8.55417)],
    }
    
    os.makedirs(OUT_DIR, exist_ok=True)
    date = pd.Timestamp(DATE)
    
    to_utm = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    to_ll = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    
    # bbox in UTM (use all four corners, a lon/lat box is not a UTM rectangle)
    cx, cy = to_utm.transform(
        [BBOX[0], BBOX[2], BBOX[0], BBOX[2]],
        [BBOX[1], BBOX[1], BBOX[3], BBOX[3]],
    )
    xmin, xmax, ymin, ymax = min(cx), max(cx), min(cy), max(cy)
    print(f"BBOX in UTM: x {xmin:.0f}-{xmax:.0f}, y {ymin:.0f}-{ymax:.0f}")
    
    
    # ---------------------------------------------------------------------------
    # 1. Find tiles and check completeness
    # ---------------------------------------------------------------------------
    def parse_name(f):
        """S1_<x_left>_<y_top>_<start>_<end>.zarr -> dict, or None."""
        m = re.search(
            r"S1_(\d+)_(\d+)_(\d{8})_(\d{8})\.zarr",
            os.path.basename(str(f).rstrip("/")),
        )
        if m is None:
            return None
        return dict(
            x_left=int(m[1]), y_top=int(m[2]),
            start=pd.Timestamp(m[3]), end=pd.Timestamp(m[4]),
        )
    
    
    files = {}   # (x_left, y_top) -> path
    for xx in np.append(np.arange(xmin, xmax, SAMPLE_STEP_M), xmax):
        for yy in np.append(np.arange(ymin, ymax, SAMPLE_STEP_M), ymax):
            lon, lat = to_ll.transform(xx, yy)
            for f in find_tiles(lat, lon):
                info = parse_name(f)
                if info and info["start"] <= date <= info["end"]:
                    files[(info["x_left"], info["y_top"])] = f
    
    if not files:
        raise SystemExit("No local tiles found for this bbox and date.")
    
    # expected tile origins on the same 1280 m grid as the found tiles
    # (tile covers x_left .. x_left+TILE_M and y_top-TILE_M .. y_top)
    ax0, ay0 = next(iter(files))
    kx = np.arange(np.floor((xmin - ax0) / TILE_M), np.floor((xmax - ax0) / TILE_M) + 1)
    ky = np.arange(np.ceil((ymin - ay0) / TILE_M), np.ceil((ymax - ay0) / TILE_M) + 1)
    expected = {(int(ax0 + i * TILE_M), int(ay0 + j * TILE_M)) for i in kx for j in ky}
    missing = sorted(expected - set(files))
    extra = sorted(set(files) - expected)
    
    print(f"\n[1] Tiles: {len(expected)} expected, {len(files)} found, {len(missing)} missing")
    for k in missing:
        print(f"    MISSING  S1_{k[0]}_{k[1]}_...")
    for k in extra:
        print(f"    extra (outside bbox grid)  S1_{k[0]}_{k[1]}_...")
    
    
    # ---------------------------------------------------------------------------
    # 2. Per-tile metadata
    # ---------------------------------------------------------------------------
    rows = []
    for (xl, yt), f in sorted(files.items()):
        z = xr.open_zarr(f, group=GROUP)
        lon, lat = z.lon.values, z.lat.values
        crs_vars = [v for v in ("spatial_ref", "crs") if v in z.variables]
        crs_attrs = [k for k in z.attrs if any(s in k.lower() for s in ("crs", "epsg", "proj"))]
        rows.append(dict(
            tile=f"{xl}_{yt}",
            nx=z.sizes["lon"], ny=z.sizes["lat"],
            res_x=float(np.median(np.diff(lon))),
            res_y=float(np.median(np.diff(lat))),
            lat_order="S->N (ascending)" if lat[0] < lat[-1] else "N->S (descending)",
            # 0 -> coords are left/top edges; RES/2 -> coords are pixel centres
            x_offset_vs_name=float(lon.min() - xl),
            y_offset_vs_name=float(yt - lat.max()),
            extent_x=float(lon.max() - lon.min() + abs(np.median(np.diff(lon)))),
            crs_metadata=", ".join(crs_vars + crs_attrs) or "NONE",
        ))
    meta = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n[2] Per-tile metadata")
    print(meta.to_string(index=False))
    print(
        "\n    x/y_offset_vs_name: 0 means the coords equal the tile edge (pixel corner"
        f"\n    convention or name = first centre); {RES/2:g} means pixel centres."
        "\n    lat_order S->N is a common cause of N-S flipped tiles in QGIS/GDAL."
        "\n    crs_metadata NONE means QGIS has to guess the CRS."
    )
    
    
    # ---------------------------------------------------------------------------
    # 3. Local mosaic for DATE on a regular grid
    # ---------------------------------------------------------------------------
    pieces = {}
    for key, f in sorted(files.items()):
        z = xr.open_zarr(f, group=GROUP)
        t = pd.to_datetime(z.time.values).normalize()
        sel = (t == date) & (z.relative_orbit.compute().values == RELATIVE_ORBIT)
        if not sel.any():
            print(f"    tile {key}: no acquisition on {DATE} for orbit {RELATIVE_ORBIT}")
            continue
        da = z.s1_vv.isel(time=sel).where(z.s1_valid.isel(time=sel) > 0)
        da = da.where(da > 0).mean("time").rename(lon="x", lat="y")
        da = da.assign_coords(x=np.round(da.x.values, 3), y=np.round(da.y.values, 3))
        pieces[key] = da.transpose("y", "x").load()
    
    if not pieces:
        raise SystemExit(f"No local data on {DATE} for orbit {RELATIVE_ORBIT}.")
    
    local = xr.concat(list(pieces.values()), dim="tile", join="outer").mean("tile", skipna=True)
    
    x_reg = np.round(np.arange(local.x.min().item(), local.x.max().item() + RES / 2, RES), 3)
    y_reg = np.round(np.arange(local.y.max().item(), local.y.min().item() - RES / 2, -RES), 3)
    local = local.reindex(x=x_reg, y=y_reg)          # north-up, gaps = NaN
    local = local.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
    
    gap_frac = float(local.isnull().mean())
    print(f"\n[3] Local mosaic: {local.sizes['x']} x {local.sizes['y']} px, "
          f"{100 * gap_frac:.1f} % NaN (gaps / invalid)")
    
    
    # ---------------------------------------------------------------------------
    # 4. PC on exactly the same pixel grid (local coords treated as pixel centres)
    # ---------------------------------------------------------------------------
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    items = catalog.search(
        collections=["sentinel-1-rtc"],
        bbox=BBOX,
        datetime=f"{DATE}/{DATE}",
        query={
            "sat:orbit_state": {"eq": ORBIT_STATE_PC},
            "sat:relative_orbit": {"eq": RELATIVE_ORBIT},
        },
    ).item_collection()
    print(f"\n[4] PC: {len(items)} item(s): {[i.id for i in items]}")
    if len(items) == 0:
        raise SystemExit("No PC items for this date / orbit.")
    
    gbox = GeoBox.from_bbox(
        (
            local.x.min().item() - RES / 2, local.y.min().item() - RES / 2,
            local.x.max().item() + RES / 2, local.y.max().item() + RES / 2,
        ),
        crs=CRS, resolution=RES, tight=True,
    )
    pc = odc.stac.load(list(items), bands=["vv"], geobox=gbox, groupby="solar_day").vv.isel(time=0)
    pc = pc.where(pc > 0).load()
    
    # compare by array position (both north-up, same grid)
    ny = min(pc.sizes["y"], local.sizes["y"])
    nx = min(pc.sizes["x"], local.sizes["x"])
    pc = pc.isel(y=slice(0, ny), x=slice(0, nx))
    local = local.isel(y=slice(0, ny), x=slice(0, nx))
    print(f"    PC grid x[0]={float(pc.x[0]):.1f}, local x[0]={float(local.x[0]):.1f}; "
          f"PC y[0]={float(pc.y[0]):.1f}, local y[0]={float(local.y[0]):.1f}")
    
    pc_db = 10 * np.log10(pc)
    loc_db = 10 * np.log10(local)
    
    
    # ---------------------------------------------------------------------------
    # 5. Agreement metrics
    # ---------------------------------------------------------------------------
    try:
        from skimage.registration import phase_cross_correlation
    except ImportError:
        phase_cross_correlation = None
        print("    (install scikit-image for shift estimates)")
    
    
    def compare(a_db, b_db):
        """Correlation, median difference and shift of b relative to a (metres)."""
        a, b = np.asarray(a_db), np.asarray(b_db)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 100:
            return dict(n=int(m.sum()), corr=np.nan, med_diff=np.nan, east=np.nan, north=np.nan)
        corr = np.corrcoef(a[m], b[m])[0, 1]
        med = float(np.median(b[m] - a[m]))
        east = north = np.nan
        if phase_cross_correlation is not None:
            af = np.where(m, a, np.nanmedian(a[m]))
            bf = np.where(m, b, np.nanmedian(b[m]))
            shift, _, _ = phase_cross_correlation(af, bf, upsample_factor=20)
            # shift = how far b must move (rows, cols) to match a
            east = -shift[1] * RES      # b is displaced this far east of a
            north = shift[0] * RES      # rows run north->south
        return dict(n=int(m.sum()), corr=corr, med_diff=med, east=east, north=north)
    
    
    overall = compare(pc_db.values, loc_db.values)
    print("\n[5] Overall (local vs PC, dB)")
    print(f"    n={overall['n']}, corr={overall['corr']:.3f}, "
          f"median(local-PC)={overall['med_diff']:+.2f} dB, "
          f"local offset={overall['east']:+.1f} m E, {overall['north']:+.1f} m N")
    
    per_tile = []
    for (xl, yt), da in pieces.items():
        xs, ys = da.x.values, da.y.values
        sub = dict(x=slice(xs.min(), xs.max()), y=slice(ys.max(), ys.min()))
        r = compare(pc_db.sel(**sub).values, loc_db.sel(**sub).values)
        r["tile"] = f"{xl}_{yt}"
        per_tile.append(r)
    per_tile = pd.DataFrame(per_tile)[["tile", "n", "corr", "med_diff", "east", "north"]]
    print("\n    Per tile:")
    print(per_tile.round(3).to_string(index=False))
    print(
        "\n    corr > ~0.7 and shift < ~2 m: tile matches PC."
        "\n    corr low for some tiles only: those tiles are misplaced or wrong data."
        f"\n    constant shift of ~{RES/2:g} m: pixel corner vs centre convention."
    )
    
    
    # ---------------------------------------------------------------------------
    # 6. Plots
    # ---------------------------------------------------------------------------
    lines_utm = {
        name: np.array([to_utm.transform(lo, la) for la, lo in pts])
        for name, pts in REF_LINES.items()
    }
    corr_by_tile = dict(zip(per_tile.tile, per_tile["corr"]))
    
    # --- 6a: tile layout ---
    fig, ax = plt.subplots(figsize=(7, 7))
    for (xl, yt) in sorted(expected | set(files)):
        found = (xl, yt) in files
        ax.add_patch(Rectangle(
            (xl, yt - TILE_M), TILE_M, TILE_M,
            fill=True, alpha=0.25, lw=1,
            facecolor="tab:green" if found else "tab:red", edgecolor="k",
        ))
        c = corr_by_tile.get(f"{xl}_{yt}", np.nan)
        label = f"{xl}\n{yt}" + (f"\nr={c:.2f}" if np.isfinite(c) else ("" if found else "\nMISSING"))
        ax.text(xl + TILE_M / 2, yt - TILE_M / 2, label, ha="center", va="center", fontsize=7)
    bx, by = to_utm.transform(
        [BBOX[0], BBOX[2], BBOX[2], BBOX[0], BBOX[0]],
        [BBOX[1], BBOX[1], BBOX[3], BBOX[3], BBOX[1]],
    )
    ax.plot(bx, by, "b-", lw=1.5, label="BBOX")
    for (name, xy), c in zip(lines_utm.items(), ["red", "cyan"]):
        ax.plot(xy[:, 0], xy[:, 1], color=c, lw=1, label=name)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.set_title("Tiles: green = found (r = corr with PC), red = missing", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/tile_layout_{DATE}.png", dpi=150)
    
    # --- 6b: PC vs local vs difference ---
    extent = [
        float(pc.x.min()) - RES / 2, float(pc.x.max()) + RES / 2,
        float(pc.y.min()) - RES / 2, float(pc.y.max()) + RES / 2,
    ]
    cmap_gray = plt.get_cmap("gray").copy()
    cmap_gray.set_bad("tab:red")              # gaps in red, not stretched
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(pc_db.values, cmap=cmap_gray, vmin=-25, vmax=0, extent=extent)
    axes[0].set_title(f"PC {DATE}", fontsize=10)
    axes[1].imshow(loc_db.values, cmap=cmap_gray, vmin=-25, vmax=0, extent=extent)
    axes[1].set_title(f"Local {DATE} (red = no data)", fontsize=10)
    im = axes[2].imshow(loc_db.values - pc_db.values, cmap="RdBu_r", vmin=-6, vmax=6, extent=extent)
    axes[2].set_title("Local - PC (dB)", fontsize=10)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    for (xl, yt) in files:                     # tile outlines
        for a in axes:
            a.add_patch(Rectangle((xl, yt - TILE_M), TILE_M, TILE_M,
                                  fill=False, ec="yellow", lw=0.6))
    for a in axes:
        for (name, xy), c in zip(lines_utm.items(), ["red", "cyan"]):
            a.plot(xy[:, 0], xy[:, 1], color=c, lw=0.8)
        a.set_xlim(extent[:2]); a.set_ylim(extent[2:])
        a.set_aspect("equal")
        a.ticklabel_format(style="plain", useOffset=False)
        a.tick_params(labelsize=7)
    
    fig.suptitle(
        f"Local vs PC, {ORBIT_STATE_PC} {RELATIVE_ORBIT}, {DATE}: "
        f"corr={overall['corr']:.2f}, median diff={overall['med_diff']:+.2f} dB, "
        f"local offset {overall['east']:+.1f} m E / {overall['north']:+.1f} m N",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/pc_vs_local_{DATE}.png", dpi=150)
    
    
    # ---------------------------------------------------------------------------
    # 7. Optional: GeoTIFFs for QGIS (north-up, CRS written)
    # ---------------------------------------------------------------------------
    try:
        import rioxarray  # noqa: F401
    
        for name, da in [("pc", pc_db), ("local", loc_db)]:
            out = f"{OUT_DIR}/{name}_vv_db_{DATE}.tif"
            da.astype("float32").rio.set_spatial_dims("x", "y").rio.write_crs(CRS).rio.to_raster(out)
            print(f"[7] wrote {out}")
        print("    Load both in QGIS: if these line up but the raw zarr does not,"
              "\n    the problem is the zarr's georeferencing metadata, not the data.")
    except ImportError:
        print("[7] install rioxarray to export GeoTIFFs for QGIS")


if __name__ == '__main__':
    start = datetime.now()
    # plot_swissimage()
    # new_test_s1()
    # compare_pc_loc()
    # full_diag()
    check_map_local()
    # plot_pixel_timeseries()
    # plot_s1_ts()
    # check_geneva_desc()
    # plot_shift_timeseries(full_timeseries=True)
    # plot_shift_timeseries_loc(full_timeseries=True)
    # for year in range(2016, 2022):
    #     analyse_year(year)
    # run_plot_centroid()
    # spatial_stable()
    # plot_spatial3locs()
    end = datetime.now()
    print(f'Calculation done at {end}. Calculation took {end-start}')