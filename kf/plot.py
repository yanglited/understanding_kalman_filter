"""Shared plotting for the examples: estimate vs. measurements, and estimate uncertainty."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def tracking_figure(truth, measurements, estimates, variances, title):
    """Two panels. Top: measurements, true value, running average, Kalman estimate with a 1-sigma band.
    Bottom: the estimate's standard deviation over time."""
    n = np.arange(1, len(measurements) + 1)
    naive_mean = np.cumsum(measurements) / n
    std = np.sqrt(variances)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.08,
                        subplot_titles=("Estimate vs. measurements", "Estimate uncertainty (std-dev)"))
    fig.add_trace(go.Scatter(x=np.r_[n, n[::-1]], y=np.r_[estimates + std, (estimates - std)[::-1]],
                             fill="toself", fillcolor="rgba(42,120,214,0.15)", line=dict(width=0),
                             name="estimate ± 1σ", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=n, y=measurements, mode="markers", name="measurements",
                             marker=dict(color="#9a9a94", size=5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=n, y=truth, mode="lines", name="true value",
                             line=dict(color="#52514e", dash="dash", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=n, y=naive_mean, mode="lines", name="running average",
                             line=dict(color="#eb6834", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=n, y=estimates, mode="lines", name="Kalman estimate",
                             line=dict(color="#2a78d6", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=n, y=std, mode="lines", name="estimate std-dev",
                             line=dict(color="#2a78d6", width=2)), row=2, col=1)
    fig.update_layout(title=title, template="plotly_white", hovermode="x unified", height=650,
                      legend=dict(orientation="h", y=-0.1))
    fig.update_xaxes(title_text="step", row=2, col=1)
    return fig


def show_or_save(fig, html=None):
    """Open the figure in a browser, or write it to an HTML file if a path is given."""
    if html:
        fig.write_html(html, include_plotlyjs="cdn")
        print("wrote", html)
    else:
        fig.show()
