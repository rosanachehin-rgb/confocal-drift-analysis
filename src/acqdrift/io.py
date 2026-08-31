"""Readers. Turn a directory of acquisitions into a tidy per-image table.

The statistical core of this package does not know what a microscope is: it
consumes a DataFrame with one row per image, a group label, an acquisition
time and some numeric metrics. Everything instrument-specific lives here, so
adding a second file format means writing a reader, not touching the science.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

# Settings that should not change during a single session. If any of these
# varies across files, drift in the image statistics has a trivial explanation
# and the rest of the audit is moot until it is resolved.
SETTING_TAGS = [
    "PinholeSizeAiry", "Voltage", "DigitalGain", "Offset", "Power",
    "ExcitationWavelength", "LaserScanPixelTime", "BitsPerPixel",
    "Attenuation", "Zoom",
]

_TIMESTAMP = re.compile(r"<AcquisitionDateAndTime>([^<]+)</AcquisitionDateAndTime>")
_CHANNEL = re.compile(r'<Channel Id="(Channel:\d+)"[^>]*Name="([^"]+)"')
_CHANNEL_BLOCK = re.compile(
    r'<Channel Id="(Channel:\d+)"[^>]*Name="([^"]+)".*?</Channel>', re.S)


def _first(xml, tag):
    hits = re.findall(rf"<{tag}[^>]*>([^<]+)</{tag}>", xml)
    return hits[0] if hits else None


def channel_offsets(xml):
    """Electronic offset added by the detector, per channel.

    Detectors commonly add a constant to every pixel so that noise does not
    clip at zero. That constant is not signal, and leaving it in the
    denominator makes every percentage smaller than it really is. On the
    worked example the offset is 100 counts against a background of 147, so
    percentages computed on the raw value understate the drift threefold.

    Rank correlations are unaffected, since adding a constant does not change
    the order of anything. Only magnitudes are.
    """
    offsets = {}
    for match in _CHANNEL_BLOCK.finditer(xml):
        name, block = match.group(2), match.group(0)
        for tag in ("DigitalOffset", "Offset"):
            found = re.findall(rf"<{tag}>([^<]+)</{tag}>", block)
            if found:
                try:
                    offsets.setdefault(name, float(found[0]))
                except ValueError:
                    pass
                break
    return offsets


def czi_metadata(path):
    """Acquisition timestamp, pixel scale, channel names and settings."""
    import czifile

    with czifile.CziFile(path) as czi:
        xml = czi.metadata()
        axes, shape = czi.axes, czi.shape

    stamp = _TIMESTAMP.search(xml)
    scaling = {}
    try:
        for dist in ET.fromstring(xml).iter("Distance"):
            axis, value = dist.get("Id"), dist.findtext("Value")
            if axis and value:
                scaling[axis] = float(value) * 1e6  # metres -> micrometres
    except ET.ParseError:
        pass

    channels = {}
    for cid, name in _CHANNEL.findall(xml):
        channels.setdefault(cid, name)
    offsets = channel_offsets(xml)

    meta = {
        "file": Path(path).name,
        "timestamp": stamp.group(1) if stamp else None,
        "px_um": scaling.get("X"),
        "z_um": scaling.get("Z"),
        "axes": axes,
        "shape": tuple(int(s) for s in shape),
        "channels": [channels[k] for k in sorted(channels)],
        "offsets": offsets,
    }
    for tag in SETTING_TAGS:
        meta[f"set_{tag}"] = _first(xml, tag)
    return meta


def czi_planes(path):
    """Yield (channel_index, volume) for every channel, as z-y-x arrays.

    Axes are looked up by name rather than assumed, because CZI files carry
    singleton H, T and sample-per-pixel axes whose position varies.
    """
    import czifile

    with czifile.CziFile(path) as czi:
        arr = czi.asarray()
        axes = czi.axes

    n_channels = arr.shape[axes.index("C")] if "C" in axes else 1
    for c in range(n_channels):
        index = []
        for axis, size in zip(axes, arr.shape):
            if axis == "C":
                index.append(c)
            elif axis in "ZYX":
                index.append(slice(None))
            else:
                index.append(0)
        vol = arr[tuple(index)]
        order = [a for a in axes if a in "ZYX"]
        vol = np.transpose(vol, [order.index(a) for a in "ZYX"])
        yield c, np.ascontiguousarray(vol)


def image_metrics(volume, bit_depth=16, offset=0.0):
    """Summary statistics of a raw volume. No segmentation, no thresholding.

    `background` is the median of every voxel. In a sparsely labelled
    preparation the objects of interest occupy a small fraction of the field,
    so the median tracks the background level and is unaffected by how many
    cells happen to be in frame. That independence is what makes it usable as
    a drift sentinel: it cannot be moved by the biology under test.
    """
    v = np.asarray(volume, dtype=np.float64).ravel()
    ceiling = float(2 ** bit_depth - 1)
    return {
        "background": float(np.median(v)),
        "p01": float(np.percentile(v, 1)),
        "p99": float(np.percentile(v, 99)),
        "p999": float(np.percentile(v, 99.9)),
        "total": float(v.sum()),
        "mad": float(np.median(np.abs(v - np.median(v)))),
        "sat_frac": float(np.mean(v >= ceiling)),
        # carried through so percentages can be taken against real signal
        "offset": float(offset),
        "n_voxels": float(v.size),
    }


def read_session(directory, group_from_name, pattern="*.czi",
                 with_pixels=True, channel_names=None):
    """Build the per-image table for a whole session.

    With `with_pixels=False` only metadata is read, which is enough for the
    schedule audit and takes about a second for a full session.
    """
    files = sorted(Path(directory).glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern} in {directory}")

    rows = []
    for path in files:
        meta = czi_metadata(path)
        row = {k: v for k, v in meta.items()
               if k not in ("channels", "shape", "axes", "offsets")}
        row["group"] = group_from_name(path.name)
        row["n_channels"] = len(meta["channels"])

        if with_pixels:
            names = channel_names or meta["channels"]
            depth = int(meta.get("set_BitsPerPixel") or 16)
            for c, vol in czi_planes(path):
                label = names[c] if c < len(names) else f"ch{c}"
                offset = meta["offsets"].get(label, 0.0)
                metrics = image_metrics(vol, bit_depth=depth, offset=offset)
                for key, value in metrics.items():
                    row[f"{label}.{key}"] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    stamps = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    df["t_min"] = (stamps - stamps.min()).dt.total_seconds() / 60.0
    return df.sort_values("t_min").reset_index(drop=True)
