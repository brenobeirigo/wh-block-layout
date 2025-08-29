
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative as qual


def plot_layout(
    df_solution: pd.DataFrame,
    building_x: float,
    building_y: float,
    *,
    title: str = "",
    label_centers: bool = True,
    label_boxes: bool = True,
    annotate_area: bool = False,
    fontsize: int = 10,
    linewidth: float = 2.0,
    cmap_name: str = "tab20",
) -> None:
    """
    Plot the building and department rectangles with distinct colors and hover tooltips.

    Parameters
    ----------
    df_solution : pandas.DataFrame
        Table indexed by department name. Must contain columns:
        ["x_west","y_south","width","height","center_x","center_y"].
        If column "area_approx" exists it will be used in the tooltip, otherwise
        area is computed as width*height.
    building_x : float
        Building size along x in meters.
    building_y : float
        Building size along y in meters.
    title : str, optional
        Plot title.
    label_centers : bool, optional
        If True, write the department name at its center.
    label_boxes : bool, optional
        If True, also draw rectangle borders on top of the fill.
    annotate_area : bool, optional
        If True, append approximate area to the center label.
    fontsize : int, optional
        Font size for labels.
    linewidth : float, optional
        Line width for rectangle borders.
    cmap_name : str, optional
        Matplotlib colormap name used to generate distinct colors.

    Returns
    -------
    None
    """
    # Figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw building envelope
    ax.add_patch(Rectangle((0.0, 0.0), building_x, building_y, fill=False, lw=1.5))
    ax.set_xlim(0.0, building_x)
    ax.set_ylim(0.0, building_y)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)

    # Build a list of colors, one per department
    n = len(df_solution)
    cmap = plt.get_cmap(cmap_name, max(n, 1))
    colors = [cmap(i % cmap.N) for i in range(n)]

    # Keep references to patches and their metadata for hover
    patches = []
    info_text = []

    for (dept, row), color in zip(df_solution.iterrows(), colors):
        # Read geometry
        try:
            x0 = float(row["x_west"]);   y0 = float(row["y_south"])
            w  = float(row["width"]);    h  = float(row["height"])
            cx = float(row["center_x"]); cy = float(row["center_y"])
        except Exception:
            continue
        if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(w) and np.isfinite(h)):
            continue

        # Fill rectangle
        face = (*color[:3], 0.35)  # same color with alpha
        edge = color
        rect = Rectangle((x0, y0), w, h, facecolor=face, edgecolor=edge, lw=linewidth)
        ax.add_patch(rect)

        # Optional border emphasis
        if label_boxes:
            rect.set_linewidth(linewidth)

        # Optional center label
        if label_centers and np.isfinite(cx) and np.isfinite(cy):
            label = str(dept)
            area_val = float(row.get("area_approx", w * h))
            if annotate_area and np.isfinite(area_val):
                label = f"{dept}\n≈ {area_val:.1f} m²"
            ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize)

        # Build tooltip text for this department
        area_val = float(row.get("area_approx", w * h))
        aspect = (max(w, h) / min(w, h)) if (w > 0 and h > 0) else np.nan
        txt = (
            f"{dept}\n"
            f"center=({cx:.2f}, {cy:.2f})\n"
            f"width×height= {w:.2f} × {h:.2f} m\n"
            f"area ≈ {area_val:.2f} m²\n"
            f"aspect ≈ {aspect:.2f}"
        )
        patches.append(rect)
        info_text.append(txt)

    # Single Annotation object used as hover tooltip
    annot = ax.annotate(
        "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),
        ha="left", va="bottom", fontsize=fontsize
    )
    annot.set_visible(False)

    def _update_annot(event, patch, text):
        """Position and set text of the tooltip near the mouse pointer."""
        annot.xy = (event.xdata, event.ydata)
        annot.set_text(text)
        annot.set_visible(True)

    def _on_move(event):
        """Mouse-move callback that shows tooltip when hovering a rectangle."""
        if not event.inaxes:
            annot.set_visible(False)
            fig.canvas.draw_idle()
            return

        # Walk patches and test if the mouse is inside
        visible_any = False
        for rect, text in zip(patches, info_text):
            contains, _ = rect.contains(event)
            if contains:
                _update_annot(event, rect, text)
                visible_any = True
                break
        if not visible_any:
            annot.set_visible(False)
        fig.canvas.draw_idle()

    # Connect hover callback
    cid = fig.canvas.mpl_connect("motion_notify_event", _on_move)

    plt.tight_layout()
    plt.show()


def _to_rgba(color: str, alpha: float) -> str | None:
    """
    Best-effort conversion of a color string to an 'rgba(r,g,b,a)' string.
    Supports '#rgb', '#rrggbb', 'rgb(r,g,b)', 'rgba(r,g,b,a)'.
    Returns None if conversion is not possible (e.g., named color).
    """
    if not isinstance(color, str):
        return None

    s = color.strip().lower()

    # Already rgba -> replace alpha
    if s.startswith("rgba"):
        nums = re.findall(r"[\d.]+", s)
        if len(nums) >= 3:
            r, g, b = [int(float(nums[0])), int(float(nums[1])), int(float(nums[2]))]
            return f"rgba({r},{g},{b},{alpha})"
        return None

    # rgb -> add alpha
    if s.startswith("rgb("):
        nums = re.findall(r"[\d.]+", s)
        if len(nums) >= 3:
            r, g, b = [int(float(nums[0])), int(float(nums[1])), int(float(nums[2]))]
            return f"rgba({r},{g},{b},{alpha})"
        return None

    # hex #rgb or #rrggbb
    if s.startswith("#"):
        h = s[1:]
        if len(h) == 3:  # short hex
            h = "".join(ch * 2 for ch in h)
        if len(h) == 6 and all(c in "0123456789abcdef" for c in h):
            r = int(h[0:2], 16)
            g = int(h[2:4], 16)
            b = int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"
        return None

    # named color (e.g., 'blue'): let Plotly handle it; we can't set alpha here
    return None


def plot_layout_plotly(
    df_solution: pd.DataFrame,
    building_x: float,
    building_y: float,
    *,
    title: str = "",
    label_centers: bool = True,
    annotate_area: bool = False,
    palette: str = "Plotly",   # try "Set3","Safe","Dark24", etc.
    opacity: float = 0.35,
    line_width: float = 2.0,
    show_legend: bool = False,
) -> go.Figure:
    """
    Interactive Plotly layout with distinct colors and hover tooltips.

    Parameters
    ----------
    df_solution : pandas.DataFrame
        Indexed by department, with columns:
        ["x_west","y_south","width","height","center_x","center_y"].
        If "area_approx" exists, it will be used in hover/labels.
    building_x, building_y : float
        Building envelope size in meters.
    title : str, optional
        Figure title.
    label_centers : bool, optional
        If True, put a text label at each department center.
    annotate_area : bool, optional
        If True, append approximate area to the center label.
    palette : str, optional
        Qualitative palette name from plotly.colors.qualitative.
    opacity : float, optional
        Fill opacity (fallback to trace opacity when rgba conversion is not possible).
    line_width : float, optional
        Department border width.
    show_legend : bool, optional
        Show legend entries for departments.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Call `fig.show()` to render.
    """
    # choose a qualitative palette
    colors = list(getattr(qual, palette, qual.Plotly))
    if not colors:
        colors = list(qual.Plotly)

    fig = go.Figure()

    # Building outline
    bx, by = building_x, building_y
    fig.add_trace(
        go.Scatter(
            x=[0, bx, bx, 0, 0],
            y=[0, 0, by, by, 0],
            mode="lines",
            line=dict(color="black", width=1.5),
            name="Building",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Departments
    for k, (dept, row) in enumerate(df_solution.iterrows()):
        try:
            x0 = float(row["x_west"]); y0 = float(row["y_south"])
            w  = float(row["width"]);  h  = float(row["height"])
            cx = float(row["center_x"]); cy = float(row["center_y"])
        except Exception:
            continue

        if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(w) and np.isfinite(h)):
            continue

        # polygon vertices (closed)
        x_poly = [x0, x0 + w, x0 + w, x0, x0]
        y_poly = [y0, y0,      y0 + h, y0 + h, y0]

        area_val = float(row.get("area_approx", w * h))
        aspect   = (max(w, h) / min(w, h)) if (w > 0 and h > 0) else np.nan

        edge_color = colors[k % len(colors)]
        fill_color = _to_rgba(edge_color, opacity)  # rgba if possible; else None

        trace_kwargs = dict(
            x=x_poly,
            y=y_poly,
            mode="lines",
            line=dict(color=edge_color, width=line_width),
            name=str(dept),
            showlegend=show_legend,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "center = (%{customdata[1]:.2f}, %{customdata[2]:.2f})<br>"
                "width × height = %{customdata[3]:.2f} × %{customdata[4]:.2f} m<br>"
                "area ≈ %{customdata[5]:.2f} m²<br>"
                "aspect ≈ %{customdata[6]:.2f}<extra></extra>"
            ),
            customdata=[[dept, cx, cy, w, h, area_val, aspect]],
            fill="toself",
        )
        if fill_color is not None:
            trace_kwargs["fillcolor"] = fill_color
        else:
            # fallback: use trace opacity if we couldn't build an rgba fill
            trace_kwargs["opacity"] = opacity

        fig.add_trace(go.Scatter(**trace_kwargs))

        # Center label
        if label_centers and np.isfinite(cx) and np.isfinite(cy):
            text = str(dept)
            if annotate_area and np.isfinite(area_val):
                text = f"{dept}<br>≈ {area_val:.1f} m²"
            fig.add_trace(
                go.Scatter(
                    x=[cx], y=[cy],
                    mode="text",
                    text=[text],
                    textposition="middle center",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # Axes with equal scaling
    fig.update_xaxes(range=[0, building_x], title="x (m)", showgrid=True, zeroline=False)
    fig.update_yaxes(range=[0, building_y], title="y (m)", showgrid=True, zeroline=False,
                     scaleanchor="x", scaleratio=1)

    fig.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h"),
        hoverlabel=dict(bgcolor="white"),
    )
    return fig