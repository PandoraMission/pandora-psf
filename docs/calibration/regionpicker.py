
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from ipywidgets import VBox, HBox, Button, Textarea, HTML, ToggleButton
from IPython.display import display

# Optional SciPy for spline refinement
try:
    from scipy.interpolate import UnivariateSpline, CubicSpline
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

class RegionPicker:
    """
    Drag to select x-ranges; picks the local minimum via cubic-spline refinement
    (fallback to quadratic). Displays only the CSV of refined x* values.
    """

    def __init__(self, x, y, title="Drag to select regions (refined min x only)",
                 window_pts=21, smoothing=None):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.window_pts = int(window_pts)
        self.smoothing = smoothing

        self.regions = []         # [(xmin, xmax)]
        self.spans = []           # axvspan patches
        self.minima = []          # [(x_refined, y_refined)]
        self.min_points = []      # refined min marker handles
        self.min_vlines = []      # vertical line handles

        # --- figure/axes
        self.fig, self.ax = plt.subplots()
        self.ax.plot(self.x, self.y)
        self.ax.set_title(title)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        # --- minimal UI
        self.info = HTML(
            value="Drag horizontally to select a region. The refined minimum x will be listed below."
        )
        self.ta_min_x = Textarea(
            value="",
            description="min x",
            placeholder="CSV of refined x* values",
            layout=dict(width="100%", height="60px")
        )

        self.btn_select = ToggleButton(value=True, description="Select (on)",
                                       tooltip="Enable/disable drawing new regions")
        self.btn_undo = Button(description="Undo", tooltip="Remove last region & marker")
        self.btn_clear = Button(description="Clear", tooltip="Remove all regions & markers")

        self.btn_select.observe(self._toggle_select, names="value")
        self.btn_undo.on_click(self.undo)
        self.btn_clear.on_click(self.clear)

        self._selector = SpanSelector(
            self.ax, onselect=self._on_span, direction="horizontal",
            useblit=True, props=dict(alpha=0.15), interactive=False, drag_from_anywhere=True
        )

        self.ui = VBox([self.info, HBox([self.btn_select, self.btn_undo, self.btn_clear]), self.ta_min_x])

    # ---------- callbacks ----------
    def _on_span(self, xmin, xmax):
        if not self.btn_select.value:
            return
        if xmin == xmax or not np.isfinite([xmin, xmax]).all():
            return
        lo, hi = sorted((float(xmin), float(xmax)))

        mask = (self.x >= lo) & (self.x <= hi) & np.isfinite(self.x) & np.isfinite(self.y)
        span_patch = self.ax.axvspan(lo, hi, alpha=0.25)
        self.spans.append(span_patch)
        self.regions.append((lo, hi))

        if not np.any(mask):
            self.minima.append((np.nan, np.nan))
            self.min_points.append(None)
            self.min_vlines.append(None)
            self._update_textbox()
            self.fig.canvas.draw_idle()
            return

        xseg = self.x[mask]
        yseg = self.y[mask]
        j = int(np.nanargmin(yseg))

        x_star, y_star = self._refine_minimum(xseg, yseg, j)

        pt, = self.ax.plot(x_star, y_star, marker="D", ms=7)  # diamond marker
        vline = self.ax.axvline(x_star, ls=":", lw=1)

        self.minima.append((x_star, y_star))
        self.min_points.append(pt)
        self.min_vlines.append(vline)

        self._update_textbox()
        self.fig.canvas.draw_idle()

    def _toggle_select(self, change):
        self.btn_select.description = "Select (on)" if change["new"] else "Select (off)"

    def _update_textbox(self):
        xs = [x for x, y in self.minima if np.isfinite(x)]
        self.ta_min_x.value = ",".join(f"{x:.6g}" for x in xs)

    # ---------- refinement core ----------
    def _refine_minimum(self, xseg, yseg, j):
        n = len(xseg)
        if n < 3:
            return float(xseg[j]), float(yseg[j])

        w = min(self.window_pts, n)
        if w % 2 == 0:
            w -= 1
        half = w // 2
        i0 = max(0, j - half)
        i1 = min(n, i0 + w)
        i0 = max(0, i1 - w)

        xw = np.asarray(xseg[i0:i1])
        yw = np.asarray(yseg[i0:i1])

        order = np.argsort(xw)
        xw = xw[order]; yw = yw[order]
        keep = np.concatenate(([True], np.diff(xw) > 0))
        xw = xw[keep]; yw = yw[keep]

        if len(xw) < 3:
            return float(xseg[j]), float(yseg[j])

        if _HAVE_SCIPY and len(xw) >= 4:
            try:
                if self.smoothing is not None:
                    spl = UnivariateSpline(xw, yw, s=self.smoothing, k=3)
                    ds = spl.derivative()
                    roots = ds.roots()
                    cands = [xw[0], xw[-1]] + [r for r in roots if xw[0] <= r <= xw[-1]]
                    vals = [(xx, float(spl(xx))) for xx in cands]
                    return min(vals, key=lambda t: t[1])
                else:
                    cs = CubicSpline(xw, yw, bc_type="not-a-knot")
                    try:
                        roots = cs.derivative().roots()
                    except Exception:
                        roots = []
                    roots = [r for r in (roots if isinstance(roots, np.ndarray) else []) if xw[0] <= r <= xw[-1]]
                    grid = np.linspace(xw[0], xw[-1], 1001)
                    cands = [xw[0], xw[-1]] + roots + list(grid)
                    yy = cs(np.array(cands))
                    kmin = int(np.nanargmin(yy))
                    return float(cands[kmin]), float(yy[kmin])
            except Exception:
                pass

        jj = np.clip(j - (i0), 1, len(xw) - 2)
        x3 = xw[jj-1:jj+2]; y3 = yw[jj-1:jj+2]
        A = np.vstack([x3**2, x3, np.ones_like(x3)]).T
        try:
            a, b, c = np.linalg.lstsq(A, y3, rcond=None)[0]
            if a != 0:
                xv = -b / (2*a)
                if xw[0] <= xv <= xw[-1]:
                    yv = a*xv**2 + b*xv + c
                    return float(xv), float(yv)
        except Exception:
            pass

        j_global = i0 + int(np.nanargmin(yw))
        return float(xseg[j_global]), float(yseg[j_global])

    # ---------- public actions ----------
    def undo(self, _=None):
        if not self.regions:
            return
        self.regions.pop()
        self.minima.pop()
        self.spans.pop().remove()
        pt = self.min_points.pop()
        if pt is not None:
            pt.remove()
        vl = self.min_vlines.pop()
        if vl is not None:
            vl.remove()
        self._update_textbox()
        self.fig.canvas.draw_idle()

    def clear(self, _=None):
        self.regions.clear()
        self.minima.clear()
        for p in self.spans:
            p.remove()
        self.spans.clear()
        for pt in self.min_points:
            if pt is not None:
                pt.remove()
        self.min_points.clear()
        for vl in self.min_vlines:
            if vl is not None:
                vl.remove()
        self.min_vlines.clear()
        self._update_textbox()
        self.fig.canvas.draw_idle()

    def widget(self):
        return self.ui
