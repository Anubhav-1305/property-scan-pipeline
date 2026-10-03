"""Render the dimensioned plan to PNG."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_plan(result, path, points=None):
    fig, ax = plt.subplots(figsize=(9, 9))
    if points is not None and len(points):
        s = points[:: max(1, len(points) // 60000)]
        ax.scatter(s[:, 0], s[:, 1], s=0.2, c="#d0d0d0", zorder=0)
    for room in result["rooms"]:
        poly = np.array(room["polygon_m"])
        closed = np.vstack([poly, poly[0]])
        ax.fill(closed[:, 0], closed[:, 1], color="#f4efe6", alpha=0.8, zorder=1)
        ax.plot(closed[:, 0], closed[:, 1], "k-", lw=2.5, zorder=3)
        for w in room["walls"]:
            if w["length_m"]["value"] < 0.8:
                continue  # skip tiny edges to keep the plan readable
            a, b = np.array(w["p0"]), np.array(w["p1"])
            mid = (a + b) / 2
            d = b - a
            n = np.array([d[1], -d[0]]) / (np.linalg.norm(d) + 1e-9)  # outward for CCW
            ax.text(*(mid + 0.28 * n), f"{w['length_m']['value']:.2f} m",
                    ha="center", va="center", fontsize=8, zorder=5)
        for o in room.get("openings", []):
            a, b = np.array(o["p0"]), np.array(o["p1"])
            col = "#2a7de1" if o["type"] == "door" else "#2aa876"
            ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=6, solid_capstyle="butt", zorder=4)
        c = poly.mean(0)
        h = room["ceiling_height_m"]
        htxt = "ceiling: not captured" if h["value"] is None else f"ceiling {h['value']:.2f} m"
        ax.text(c[0], c[1], f"{room['name']}\n{room['floor_area_m2']['value']:.1f} m²\n{htxt}",
                ha="center", va="center", fontsize=10, weight="bold", zorder=6)
    ax.set_aspect("equal")
    ax.set_title(result.get("title", "Floor plan"))
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
