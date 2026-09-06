"""Build HEC-RAS channel geometry from the Saitama river point cloud.

Pipeline
--------
1. Order tiles along the river to form a coarse centreline.
2. Resample the centreline at a fixed spacing to place cross-section stations.
3. At each station, cut a line normal to the centreline and gather nearby points.
4. Reduce those points to a station/elevation profile.

Step 4 is where the care is needed. The published tiles carry **no ground
classification** - every point is class 1 - so there is no ``classification == 2``
filter to select bare earth. Vegetation, bridges and structures sit in the same
cloud as the bed.

The approach here bins points by offset across the section and takes a low
quantile of elevation in each bin. The minimum would be the obvious choice, but
it latches onto noise and multibeam outliers; a low quantile keeps the bed while
rejecting both the canopy above and stray returns below. ``bed_quantile`` is
exposed so this can be tuned against surveyed sections.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aquanexus.logger import get_logger

log = get_logger("hecras.geometry")


@dataclass
class CrossSection:
    """A single cross-section profile, in HEC-RAS terms.

    ``station`` runs left-to-right across the section (metres from the left end);
    ``elevation`` is the derived bed level at that station. ``river_station`` is
    the distance downstream along the centreline, which HEC-RAS uses to order
    sections within a reach.
    """

    river_station: float
    station: np.ndarray
    elevation: np.ndarray
    origin: tuple[float, float]
    direction: tuple[float, float]
    n_points: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.station.shape != self.elevation.shape:
            raise ValueError("station and elevation must have matching shape")

    @property
    def is_empty(self) -> bool:
        return self.station.size == 0

    @property
    def thalweg(self) -> float:
        """Lowest bed elevation in the section - the channel invert."""
        if self.is_empty:
            return float("nan")
        return float(np.min(self.elevation))

    @property
    def width(self) -> float:
        if self.is_empty:
            return 0.0
        return float(self.station.max() - self.station.min())


# ---------------------------------------------------------------------------
# Centreline
# ---------------------------------------------------------------------------


def resample_centreline(vertices: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample a polyline at uniform spacing.

    Returns ``(points, tangents)`` where points is (M, 2) and tangents is (M, 2)
    of unit vectors. Cross-sections are cut normal to the tangent.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("vertices must be an (N, 2) array")
    if len(vertices) < 2:
        raise ValueError("need at least two vertices to define a centreline")
    if spacing <= 0:
        raise ValueError("spacing must be positive")

    seg = np.diff(vertices, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    keep = seg_len > 0
    if not keep.any():
        raise ValueError("centreline has zero length")
    seg, seg_len = seg[keep], seg_len[keep]
    vertices = np.vstack((vertices[0], vertices[1:][keep]))

    cumulative = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = cumulative[-1]
    targets = np.arange(0.0, total + 1e-9, spacing)

    points = np.empty((len(targets), 2))
    tangents = np.empty((len(targets), 2))
    for i, dist in enumerate(targets):
        j = min(int(np.searchsorted(cumulative, dist, side="right") - 1), len(seg) - 1)
        frac = (dist - cumulative[j]) / seg_len[j]
        points[i] = vertices[j] + frac * seg[j]
        tangents[i] = seg[j] / seg_len[j]

    return points, tangents


def smooth_centreline(vertices: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving-average smoother.

    Tile centroids zig-zag around the true channel axis, and that jitter rotates
    the section normals - which matters more than the positional error, because a
    section cut at the wrong angle overstates channel width.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if window <= 1 or len(vertices) < window:
        return vertices
    if window % 2 == 0:
        window += 1

    pad = window // 2
    padded = np.vstack(
        (np.repeat(vertices[:1], pad, axis=0), vertices, np.repeat(vertices[-1:], pad, axis=0))
    )
    kernel = np.ones(window) / window
    return np.column_stack(
        [np.convolve(padded[:, d], kernel, mode="valid") for d in range(2)]
    )


# ---------------------------------------------------------------------------
# Cross-section extraction
# ---------------------------------------------------------------------------


def extract_cross_section(
    points: np.ndarray,
    origin: tuple[float, float] | np.ndarray,
    tangent: tuple[float, float] | np.ndarray,
    river_station: float = 0.0,
    half_width: float = 60.0,
    strip_thickness: float = 2.0,
    bin_width: float = 1.0,
    bed_quantile: float = 0.05,
    min_points_per_bin: int = 3,
) -> CrossSection:
    """Cut one cross-section out of a point cloud.

    Points within ``strip_thickness`` of the section line and ``half_width`` of
    the centreline are binned by cross-stream offset; each bin contributes one
    bed elevation at the ``bed_quantile`` of its elevations.

    Bins holding fewer than ``min_points_per_bin`` points are dropped, since a
    quantile over one or two returns is meaningless and those are usually
    isolated noise beyond the bank.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an (N, 3) array of x, y, z")
    if not 0.0 <= bed_quantile <= 1.0:
        raise ValueError("bed_quantile must lie in [0, 1]")

    origin = np.asarray(origin, dtype=np.float64)
    tangent = np.asarray(tangent, dtype=np.float64)
    norm = np.hypot(*tangent)
    if norm == 0:
        raise ValueError("tangent must be non-zero")
    tangent = tangent / norm
    normal = np.array([-tangent[1], tangent[0]])  # left-hand normal

    rel = points[:, :2] - origin
    along = rel @ tangent      # distance up/downstream of the section line
    across = rel @ normal      # offset across the channel

    mask = (np.abs(along) <= strip_thickness) & (np.abs(across) <= half_width)
    if not mask.any():
        log.debug("no points near station %.1f", river_station)
        return CrossSection(river_station, np.empty(0), np.empty(0),
                            tuple(origin), tuple(tangent), 0)

    across_sel = across[mask]
    z_sel = points[mask, 2]

    edges = np.arange(-half_width, half_width + bin_width, bin_width)
    idx = np.clip(np.digitize(across_sel, edges) - 1, 0, len(edges) - 2)

    stations: list[float] = []
    elevations: list[float] = []
    for b in range(len(edges) - 1):
        in_bin = idx == b
        count = int(in_bin.sum())
        if count < min_points_per_bin:
            continue
        stations.append(float((edges[b] + edges[b + 1]) / 2.0))
        elevations.append(float(np.quantile(z_sel[in_bin], bed_quantile)))

    # HEC-RAS expects stationing to increase left-to-right.
    station_arr = np.asarray(stations) + half_width
    order = np.argsort(station_arr)

    return CrossSection(
        river_station=river_station,
        station=station_arr[order],
        elevation=np.asarray(elevations)[order],
        origin=(float(origin[0]), float(origin[1])),
        direction=(float(tangent[0]), float(tangent[1])),
        n_points=int(mask.sum()),
        metadata={"bed_quantile": bed_quantile, "bin_width": bin_width},
    )


def extract_sections(
    points: np.ndarray,
    centreline: np.ndarray,
    spacing: float = 500.0,
    smooth_window: int = 5,
    **kwargs,
) -> list[CrossSection]:
    """Extract cross-sections at uniform spacing along a centreline.

    Empty sections are dropped: near the ends of a reach the centreline can run
    past the point cloud.
    """
    line = smooth_centreline(centreline, smooth_window)
    origins, tangents = resample_centreline(line, spacing)

    sections = []
    for i, (origin, tangent) in enumerate(zip(origins, tangents, strict=True)):
        xs = extract_cross_section(
            points, origin, tangent, river_station=i * spacing, **kwargs
        )
        if not xs.is_empty:
            sections.append(xs)

    log.info("extracted %d of %d candidate sections", len(sections), len(origins))
    return sections


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def bank_stations(xs: CrossSection, height_fraction: float = 0.5) -> tuple[float, float]:
    """Locate the main-channel bank stations for a section.

    Walks outward from the thalweg while the bed stays below
    ``invert + height_fraction * (top - invert)``, and returns the station values
    at which it stops.

    Returning **actual station values** is a hard HEC-RAS requirement, not a
    nicety. Interpolated banks - the obvious "35% of the way across" rule - are
    rejected with::

        - Left bank station not in station elevation data.
        - Right bank station not in station elevation data.

    and the run is refused. Extraction drops sparse bins, so the station list has
    gaps and an arithmetic guess frequently lands in one.
    """
    if xs.is_empty:
        return 0.0, 0.0
    if len(xs.station) < 3:
        return float(xs.station[0]), float(xs.station[-1])

    elevation = xs.elevation
    invert_index = int(np.argmin(elevation))
    invert = float(elevation[invert_index])
    top = float(elevation.max())
    threshold = invert + height_fraction * (top - invert)

    left = invert_index
    while left > 0 and elevation[left - 1] <= threshold:
        left -= 1
    right = invert_index
    while right < len(elevation) - 1 and elevation[right + 1] <= threshold:
        right += 1

    # A flat or noisy profile can collapse both walks onto the thalweg; fall back
    # to the section ends so the banks still bracket a channel.
    if left == right:
        left, right = 0, len(elevation) - 1

    return float(xs.station[left]), float(xs.station[right])


def to_hecras_geometry(
    sections: list[CrossSection],
    title: str = "AquaNexus",
    river: str = "Ayase",
    reach: str = "Main",
    version: str = "7.00",
    manning_channel: float = 0.035,
    manning_overbank: float = 0.06,
    bank_fraction: float = 0.5,
    edited_time: str = "Jan-01-2026 00:00:00",
    gis_cut_lines: bool = False,
) -> str:
    """Serialise sections to HEC-RAS .g01 geometry format.

    Verified against HEC-RAS 7.0 via the COM controller: the emitted file loads
    with the expected river, reach and cross-section stations. Several elements
    are not optional even though they look cosmetic, and omitting them makes
    HEC-RAS silently report zero rivers rather than raise an error:

    * river and reach names are **fixed-width, padded to 16 characters**
    * ``Reach XY`` must carry the reach centreline, 16-character fields, two
      points per line
    * every cross-section needs ``XS GIS Cut Line``, ``#Mann``, ``Bank Sta``,
      ``XS Rating Curve`` and ``Exp/Cntr``
    * station/elevation pairs are 8-character fields, five pairs per line

    Sections are written in descending river station, as HEC-RAS requires.
    """
    ordered = sorted(sections, key=lambda s: s.river_station, reverse=True)

    def f8(v: float) -> str:
        return f"{v:8.2f}"

    def f16(v: float) -> str:
        return f"{v:16.4f}"

    lines = [f"Geom Title={title}", f"Program Version={version}"]

    if ordered:
        xs_all = [s.origin[0] for s in ordered]
        ys_all = [s.origin[1] for s in ordered]
        pad = 500.0
        lines.append(
            f"Viewing Rectangle= {min(xs_all) - pad:.0f} , {max(xs_all) + pad:.0f} , "
            f"{min(ys_all) - pad:.0f} , {max(ys_all) + pad:.0f}"
        )
    lines.append("")
    lines.append(f"River Reach={river:<16},{reach:<16}")

    if ordered:
        lines.append(f"Reach XY= {len(ordered)} ")
        row: list[str] = []
        for s in ordered:
            row += [f16(s.origin[0]), f16(s.origin[1])]
            if len(row) == 4:
                lines.append("".join(row))
                row = []
        if row:
            lines.append("".join(row))
        cx = sum(xs_all) / len(xs_all)
        cy = sum(ys_all) / len(ys_all)
        lines.append(f"Rch Text X Y={cx},{cy},0,0")
    lines.append("Reverse River Text= 0 ")
    lines.append("")

    for i, xs in enumerate(ordered):
        if xs.is_empty:
            continue
        # Downstream reach lengths: distance to the next section downstream.
        nxt = 0.0 if i == len(ordered) - 1 else abs(
            ordered[i + 1].river_station - xs.river_station
        )
        lines.append(
            f"Type RM Length L Ch R = 1 ,{xs.river_station:<8.0f},{nxt:.0f},{nxt:.0f},{nxt:.0f}"
        )

        # Present in files HEC-RAS writes itself; absent, the section is
        # treated as incomplete.
        lines.append(f"Node Last Edited Time={edited_time}")

        # Georeferenced cut lines are optional and off by default: a working
        # reference project written by HEC-RAS 7.0 omits them entirely, so they
        # are one more thing to get wrong before the model will run. Enable once
        # the geometry is needed in RAS Mapper.
        if gis_cut_lines:
            ox, oy = xs.origin
            dx, dy = xs.direction
            nx, ny = -dy, dx
            half = xs.width / 2.0 if xs.width else 1.0
            lines.append("XS GIS Cut Line=2")
            lines.append(
                f16(ox - nx * half) + f16(oy - ny * half)
                + f16(ox + nx * half) + f16(oy + ny * half)
            )

        n = len(xs.station)
        lines.append(f"#Sta/Elev= {n} ")
        cells = [f8(st) + f8(el) for st, el in zip(xs.station, xs.elevation, strict=True)]
        for k in range(0, len(cells), 5):
            lines.append("".join(cells[k : k + 5]))

        left_bank, right_bank = bank_stations(xs, bank_fraction)
        lines.append("#Mann= 3 ,0,0")
        lines.append(
            f8(float(xs.station.min())) + f8(manning_overbank) + f8(0)
            + f8(left_bank) + f8(manning_channel) + f8(0)
            + f8(right_bank) + f8(manning_overbank) + f8(0)
        )
        lines.append(f"Bank Sta={left_bank:.2f},{right_bank:.2f}")
        lines.append("XS Rating Curve= 0 ,0")

        # Hydraulic-table bounds. Unused by a steady run (the plan sets
        # Run HTab= 0), but HEC-RAS writes them for every section and treats
        # geometry that omits them as incomplete.
        invert = xs.thalweg
        top = float(xs.elevation.max())
        increment = max((top - invert) / 20.0, 0.01)
        lines.append(
            f"XS HTab Starting El and Incr={invert + 0.15:.2f},{increment:.2f}, 20 "
        )
        lines.append("XS HTab Horizontal Distribution= 5 , 5 , 5 ")
        lines.append("Exp/Cntr=0.3,0.1")
        lines.append("")

    # File trailer. Every one of these appears in geometry HEC-RAS writes itself.
    lines += [
        "LCMann Time=Dec-30-1899 00:00:00",
        "LCMann Region Time=Dec-30-1899 00:00:00",
        "LCMann Table=0",
        "Chan Stop Cuts=-1 ",
        "",
        "",
        "",
        "Use User Specified Reach Order=0",
        "GIS Ratio Cuts To Invert=-1",
        "GIS Limit At Bridges=0",
        "Composite Channel Slope=5",
    ]

    return "\n".join(lines) + "\n"
