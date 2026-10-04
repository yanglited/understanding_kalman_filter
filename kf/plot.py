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


def _pdf(x, mean, var):
    return np.exp(-0.5 * (x - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)


def _belief_traces(x, n, prior, measurement, measurement_var, posterior, previous=None):
    """The curves for one step: (optional) previous posterior, prior, measurement likelihood, posterior."""
    import plotly.graph_objects as go
    traces = []
    if previous is not None:
        traces.append(go.Scatter(x=x, y=_pdf(x, *previous), name="previous posterior", line=dict(color="#9a9a94", dash="dot", width=1.5)))
    traces += [
        go.Scatter(x=x, y=_pdf(x, *prior), name="prior", line=dict(color="#eb6834", width=2)),
        go.Scatter(x=x, y=_pdf(x, measurement, measurement_var), name="measurement", line=dict(color="#9a9a94", width=2)),
        go.Scatter(x=x, y=_pdf(x, *posterior), name="posterior", line=dict(color="#2a78d6", width=2.5), fill="tozeroy", fillcolor="rgba(42,120,214,0.15)"),
    ]
    return traces


def _window(n, truth, measurements, measurement_var, prior_means, prior_vars, post_means, post_vars, points=200):
    """x grid for step n: wide enough to hold every curve of that step."""
    pts = [truth[n], measurements[n], prior_means[n], post_means[n]]
    spread = 2.5 * max(np.sqrt(prior_vars[n]), np.sqrt(measurement_var))
    return np.linspace(min(pts) - spread, max(pts) + spread, points, dtype=np.float32)


def _belief_title(n, prior, measurement_var, posterior):
    return (f"step {n + 1}:  prior σ = {np.sqrt(prior[1]):.2f}   measurement σ = {np.sqrt(measurement_var):.2f}"
            f"   posterior σ = {np.sqrt(posterior[1]):.2f}")


def belief_animation(truth, measurements, measurement_var, prior_means, prior_vars, post_means, post_vars,
                     title, show_previous=False):
    """One frame per step with a slider: how the prior and the measurement combine into the posterior."""
    import plotly.graph_objects as go
    args = (truth, measurements, measurement_var, prior_means, prior_vars, post_means, post_vars)

    def step(n):
        x = _window(n, *args)
        prev = (post_means[n - 1], post_vars[n - 1]) if show_previous and n > 0 else (
            (prior_means[0], prior_vars[0]) if show_previous else None)
        tr = _belief_traces(x, n, (prior_means[n], prior_vars[n]), measurements[n], measurement_var,
                            (post_means[n], post_vars[n]), prev)
        ymax = 1.1 * max(t.y.max() for t in tr)
        return tr, ymax

    frames = []
    for n in range(len(measurements)):
        tr, ymax = step(n)
        for t in tr:
            t.update(y=t.y.astype(np.float32))
        frames.append(go.Frame(data=tr, name=str(n), layout=dict(
            yaxis=dict(range=[0, ymax]), xaxis=dict(range=[float(tr[0].x[0]), float(tr[0].x[-1])]), title=dict(text=f"{title}<br><sup>{_belief_title(n, (prior_means[n], prior_vars[n]), measurement_var, (post_means[n], post_vars[n]))}</sup>"),
            shapes=[dict(type="line", x0=truth[n], x1=truth[n], y0=0, y1=1, yref="paper", line=dict(color="#52514e", dash="dash", width=1.5))])))
    tr0, ymax0 = step(0)
    fig = go.Figure(data=tr0, frames=frames, layout=frames[0].layout)
    fig.update_layout(template="plotly_white", height=550, xaxis_title="value", yaxis_title="probability density",
                      legend=dict(orientation="h", y=-0.18),
                      annotations=[dict(text="dashed line: true value", xref="paper", yref="paper", x=1, y=1.02, showarrow=False, font=dict(color="#52514e", size=11))],
                      updatemenus=[dict(type="buttons", x=0, y=-0.3, xanchor="left", buttons=[
                          dict(label="▶ play", method="animate", args=[None, dict(frame=dict(duration=350, redraw=True), fromcurrent=True)]),
                          dict(label="❚❚ pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
                      sliders=[dict(x=0.15, len=0.85, y=-0.3, currentvalue=dict(prefix="step "), steps=[
                          dict(label=str(n + 1), method="animate", args=[[str(n)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                          for n in range(len(frames))])])
    return fig


def belief_grid(truth, measurements, measurement_var, prior_means, prior_vars, post_means, post_vars,
                title, steps, show_previous=False):
    """Static small multiples of selected steps (for the README)."""
    from plotly.subplots import make_subplots
    args = (truth, measurements, measurement_var, prior_means, prior_vars, post_means, post_vars)
    seen = set()
    fig = make_subplots(rows=2, cols=2, subplot_titles=[_belief_title(n, (prior_means[n], prior_vars[n]), measurement_var, (post_means[n], post_vars[n])) for n in steps],
                        horizontal_spacing=0.06, vertical_spacing=0.16)
    for i, n in enumerate(steps):
        r, c = divmod(i, 2)
        prev = (post_means[n - 1], post_vars[n - 1]) if show_previous and n > 0 else None
        x = _window(n, *args)
        for t in _belief_traces(x, n, (prior_means[n], prior_vars[n]), measurements[n], measurement_var, (post_means[n], post_vars[n]), prev):
            t.showlegend = t.name not in seen
            seen.add(t.name)
            fig.add_trace(t, row=r + 1, col=c + 1)
        fig.add_vline(x=truth[n], line=dict(color="#52514e", dash="dash", width=1.5), row=r + 1, col=c + 1)
    fig.update_layout(title=title, template="plotly_white", height=650, legend=dict(orientation="h", y=-0.08))
    fig.update_annotations(font_size=12)
    return fig
