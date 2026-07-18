from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display
from ipywidgets import (
    Button,
    Checkbox,
    Dropdown,
    GridBox,
    HTML,
    HBox,
    IntText,
    Layout,
    Output,
    SelectMultiple,
    ToggleButtons,
    VBox,
)
from plotly.subplots import make_subplots


@dataclass(frozen=True)
class DashboardSelection:
    model_group: str
    model_labels: tuple[str, ...]
    dataset: str
    horizon: int
    view_mode: str
    log_y: bool
    show_rolling_error: bool
    rolling_window: int


class PredictionPlotlyDashboard:
    """Interactive Plotly dashboard for prediction time-series diagnostics.

    The dashboard never writes figures or output assets to disk. Figures are
    generated only by explicit button callbacks or direct method calls.
    """

    REQUIRED_COLUMNS = {
        "dataset",
        "branch",
        "tier",
        "model",
        "time",
        "horizon",
        "true_volatility",
        "predict_volatility",
    }

    VIEW_MODES = ("Single model", "Overlay models", "Error diagnostics", "Outliers")

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        default_dataset: str | None = None,
        default_horizon: int | None = None,
        default_group: str | None = None,
        default_model: str | None = None,
        rolling_window: int = 20,
        top_outliers: int = 30,
    ) -> None:
        self.df = self.prepare_data(df)
        self.top_outliers = top_outliers
        self.summary = self._summary_html()
        self.controls = self._build_controls(
            default_dataset=default_dataset,
            default_horizon=default_horizon,
            default_group=default_group,
            default_model=default_model,
            rolling_window=rolling_window,
        )
        self.output = Output()
        self.widget = self._compose_widget()

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs) -> "PredictionPlotlyDashboard":
        df = pd.read_csv(path, low_memory=False)
        return cls(df, **kwargs)

    @classmethod
    def prepare_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(cls.REQUIRED_COLUMNS.difference(df.columns))
        if missing:
            raise ValueError(f"Prediction data is missing required columns: {missing}")

        out = df.copy()
        out["time"] = pd.to_datetime(out["time"], errors="coerce")
        out["branch"] = out["branch"].fillna("Unknown").astype(str)
        out["model"] = out["model"].fillna("Unknown").astype(str)
        out["tier"] = (
            out["tier"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace({"": "No Tier", "None": "No Tier", "nan": "No Tier", "NaN": "No Tier"})
        )
        out["horizon"] = pd.to_numeric(out["horizon"], errors="coerce").astype("Int64")
        out["true_volatility"] = pd.to_numeric(out["true_volatility"], errors="coerce")
        out["predict_volatility"] = pd.to_numeric(out["predict_volatility"], errors="coerce")
        out["model_group"] = out["branch"] + " / " + out["tier"]
        out["model_label"] = out["model_group"] + " / " + out["model"]
        out["abs_error"] = (out["true_volatility"] - out["predict_volatility"]).abs()
        denom = out["true_volatility"].abs().clip(lower=1e-8)
        out["relative_error"] = out["abs_error"] / denom
        out["squared_error"] = (out["true_volatility"] - out["predict_volatility"]) ** 2
        return out.sort_values(["dataset", "horizon", "model_group", "model_label", "time"]).reset_index(drop=True)

    def display(self) -> VBox:
        return self.widget

    def current_selection(self) -> DashboardSelection:
        return DashboardSelection(
            model_group=self.controls["group"].value,
            model_labels=tuple(self.controls["models"].value),
            dataset=self.controls["dataset"].value,
            horizon=int(self.controls["horizon"].value),
            view_mode=self.controls["view_mode"].value,
            log_y=bool(self.controls["log_y"].value),
            show_rolling_error=bool(self.controls["rolling_error"].value),
            rolling_window=max(1, int(self.controls["rolling_window"].value)),
        )

    def render_current(self) -> None:
        selection = self.current_selection()
        with self.output:
            self.output.clear_output(wait=True)
            try:
                if selection.view_mode == "Outliers":
                    display(self._styled_outlier_table(selection))
                else:
                    display(self.figure_for_selection(selection))
                self._set_status(selection)
            except Exception as exc:
                self.controls["status"].value = self._status_html(f"Render failed: {exc}", tone="error")
                print(f"Unable to render dashboard view: {exc}")

    def clear(self) -> None:
        self.output.clear_output(wait=True)
        self.controls["status"].value = self._status_html("Canvas cleared", tone="muted")

    def figure_for_selection(self, selection: DashboardSelection | None = None) -> go.Figure:
        selection = selection or self.current_selection()
        if selection.view_mode == "Single model":
            return self.single_model_figure(selection)
        if selection.view_mode == "Overlay models":
            return self.overlay_figure(selection)
        if selection.view_mode == "Error diagnostics":
            return self.error_diagnostics_figure(selection)
        raise ValueError(f"{selection.view_mode!r} is a table view, not a figure view")

    def outlier_table(self, selection: DashboardSelection | None = None) -> pd.DataFrame:
        selection = selection or self.current_selection()
        data = self._filtered_rows(selection, require_models=False)
        if selection.model_labels:
            data = data[data["model_label"].isin(selection.model_labels)]
        columns = [
            "dataset",
            "horizon",
            "model_group",
            "model_label",
            "time",
            "true_volatility",
            "predict_volatility",
            "abs_error",
            "relative_error",
            "squared_error",
        ]
        valid = data.dropna(subset=["abs_error", "predict_volatility"])
        return valid.sort_values(["abs_error", "predict_volatility"], ascending=False).loc[:, columns].head(
            self.top_outliers
        )

    def single_model_figure(self, selection: DashboardSelection) -> go.Figure:
        model_label = self._first_model(selection)
        data = self._filtered_rows(selection, model_labels=[model_label]).dropna(
            subset=["time", "true_volatility", "predict_volatility"]
        )
        if data.empty:
            raise ValueError("No valid rows for the selected single-model view")

        if selection.show_rolling_error:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
        else:
            fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data["time"],
                y=data["true_volatility"],
                mode="lines",
                name="True volatility",
                line={"color": "#1f2937", "width": 2},
            ),
            secondary_y=False if selection.show_rolling_error else None,
        )
        fig.add_trace(
            go.Scatter(
                x=data["time"],
                y=data["predict_volatility"],
                mode="lines",
                name="Predicted volatility",
                line={"color": "#d97706", "width": 1.6},
            ),
            secondary_y=False if selection.show_rolling_error else None,
        )

        if selection.show_rolling_error:
            rolling = data["abs_error"].rolling(selection.rolling_window, min_periods=1).mean()
            fig.add_trace(
                go.Scatter(
                    x=data["time"],
                    y=rolling,
                    mode="lines",
                    name=f"Rolling abs error ({selection.rolling_window})",
                    line={"color": "#2563eb", "width": 1.4, "dash": "dot"},
                ),
                secondary_y=True,
            )
            fig.update_yaxes(title_text="Rolling absolute error", secondary_y=True)

        self._apply_common_layout(
            fig,
            title=f"{selection.dataset} | h={selection.horizon} | {model_label}",
            y_title="Volatility",
            log_y=selection.log_y,
            has_secondary_y=selection.show_rolling_error,
        )
        return fig

    def overlay_figure(self, selection: DashboardSelection) -> go.Figure:
        model_labels = self._selected_or_all_group_models(selection)
        data = self._filtered_rows(selection, model_labels=model_labels).dropna(
            subset=["time", "true_volatility", "predict_volatility"]
        )
        if data.empty:
            raise ValueError("No valid rows for the selected overlay view")

        fig = go.Figure()
        true_series = data.groupby("time", as_index=False)["true_volatility"].median()
        fig.add_trace(
            go.Scatter(
                x=true_series["time"],
                y=true_series["true_volatility"],
                mode="lines",
                name="True volatility",
                line={"color": "#111827", "width": 2.4},
            )
        )
        for label, group in data.groupby("model_label", sort=True):
            fig.add_trace(
                go.Scatter(
                    x=group["time"],
                    y=group["predict_volatility"],
                    mode="lines",
                    name=label,
                    line={"width": 1.2},
                    opacity=0.78,
                )
            )
        self._apply_common_layout(
            fig,
            title=f"Prediction overlay | {selection.dataset} | h={selection.horizon}",
            y_title="Volatility",
            log_y=selection.log_y,
            has_secondary_y=False,
        )
        return fig

    def error_diagnostics_figure(self, selection: DashboardSelection) -> go.Figure:
        model_labels = self._selected_or_all_group_models(selection)
        data = self._filtered_rows(selection, model_labels=model_labels).dropna(subset=["time", "abs_error"])
        if data.empty:
            raise ValueError("No valid rows for the selected error diagnostics view")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for label, group in data.groupby("model_label", sort=True):
            fig.add_trace(
                go.Scatter(
                    x=group["time"],
                    y=group["abs_error"],
                    mode="lines",
                    name=f"{label} abs",
                    line={"width": 1.2},
                ),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=group["time"],
                    y=group["relative_error"],
                    mode="lines",
                    name=f"{label} relative",
                    line={"width": 1.0, "dash": "dot"},
                    opacity=0.6,
                ),
                secondary_y=True,
            )
        fig.update_yaxes(title_text="Absolute error", secondary_y=False)
        fig.update_yaxes(title_text="Relative error", secondary_y=True)
        self._apply_common_layout(
            fig,
            title=f"Error diagnostics | {selection.dataset} | h={selection.horizon}",
            y_title="Absolute error",
            log_y=False,
            has_secondary_y=True,
        )
        return fig

    def _build_controls(
        self,
        *,
        default_dataset: str | None,
        default_horizon: int | None,
        default_group: str | None,
        default_model: str | None,
        rolling_window: int,
    ) -> dict[str, object]:
        groups = sorted(self.df["model_group"].dropna().unique().tolist())
        datasets = sorted(self.df["dataset"].dropna().unique().tolist())
        horizons = sorted(int(v) for v in self.df["horizon"].dropna().unique().tolist())

        selected_group = default_group if default_group in groups else (groups[0] if groups else None)
        model_options = self._models_for_group(selected_group)
        selected_model = default_model if default_model in model_options else (model_options[0] if model_options else None)

        controls: dict[str, object] = {
            "title": HTML(
                value=(
                    "<div style='font:600 20px/1.2 Inter,Segoe UI,Arial,sans-serif;color:#172033;'>"
                    "Prediction Diagnostics</div>"
                    "<div style='font:13px/1.4 Inter,Segoe UI,Arial,sans-serif;color:#64748b;margin-top:4px;'>"
                    "Interactive volatility forecast review</div>"
                )
            ),
            "summary": HTML(value=self.summary),
            "status": HTML(value=self._status_html("Select filters and render a view", tone="muted")),
            "group": Dropdown(
                options=groups,
                value=selected_group,
                description="Group",
                style={"description_width": "64px"},
                layout=Layout(width="100%"),
            ),
            "models": SelectMultiple(
                options=model_options,
                value=(selected_model,) if selected_model else (),
                description="Models",
                rows=min(max(len(model_options), 4), 12),
                style={"description_width": "64px"},
                layout=Layout(width="100%", min_height="118px"),
            ),
            "dataset": Dropdown(
                options=datasets,
                value=default_dataset if default_dataset in datasets else (datasets[0] if datasets else None),
                description="Dataset",
                style={"description_width": "64px"},
                layout=Layout(width="100%"),
            ),
            "horizon": Dropdown(
                options=horizons,
                value=default_horizon if default_horizon in horizons else (horizons[0] if horizons else None),
                description="Horizon",
                style={"description_width": "64px"},
                layout=Layout(width="100%"),
            ),
            "view_mode": ToggleButtons(
                options=self.VIEW_MODES,
                value="Single model",
                description="View",
                style={"description_width": "64px", "button_width": "150px"},
                layout=Layout(width="100%"),
            ),
            "log_y": Checkbox(value=False, description="Log Y", indent=False, layout=Layout(width="90px")),
            "rolling_error": Checkbox(value=False, description="Rolling error", indent=False, layout=Layout(width="140px")),
            "rolling_window": IntText(
                value=max(1, int(rolling_window)),
                description="Window",
                style={"description_width": "64px"},
                layout=Layout(width="150px"),
            ),
            "render": Button(
                description="Render",
                button_style="primary",
                icon="line-chart",
                layout=Layout(width="132px", height="36px"),
            ),
            "clear": Button(description="Clear", icon="eraser", layout=Layout(width="112px", height="36px")),
            "show_outliers": Button(
                description="Outliers",
                button_style="warning",
                icon="exclamation-triangle",
                layout=Layout(width="128px", height="36px"),
            ),
        }

        controls["group"].observe(self._on_group_change, names="value")
        controls["render"].on_click(lambda _: self.render_current())
        controls["clear"].on_click(lambda _: self.clear())
        controls["show_outliers"].on_click(self._on_show_outliers)
        return controls

    def _compose_widget(self) -> VBox:
        top_filters = GridBox(
            children=[self.controls["group"], self.controls["dataset"], self.controls["horizon"]],
            layout=Layout(
                width="100%",
                grid_template_columns="minmax(280px, 1.4fr) minmax(220px, 1fr) minmax(160px, 0.7fr)",
                grid_gap="12px",
                align_items="center",
            ),
        )
        options_row = HBox(
            [self.controls["log_y"], self.controls["rolling_error"], self.controls["rolling_window"]],
            layout=Layout(gap="16px", align_items="center", justify_content="flex-start", flex_flow="row wrap"),
        )
        action_row = HBox(
            [self.controls["render"], self.controls["show_outliers"], self.controls["clear"], self.controls["status"]],
            layout=Layout(gap="10px", align_items="center", flex_flow="row wrap"),
        )
        control_panel = VBox(
            [
                HBox(
                    [self.controls["title"], self.controls["summary"]],
                    layout=Layout(justify_content="space-between", align_items="flex-start", gap="16px"),
                ),
                top_filters,
                self.controls["view_mode"],
                options_row,
                self.controls["models"],
                action_row,
            ],
            layout=Layout(
                width="100%",
                padding="18px 20px",
                border="1px solid #d8dee9",
                margin="0 0 14px 0",
            ),
        )
        self.output.layout = Layout(
            width="100%",
            min_height="520px",
            border="1px solid #e2e8f0",
            padding="8px 10px",
        )
        return VBox(
            [control_panel, self.output],
            layout=Layout(width="100%", max_width="1180px", margin="0 auto"),
        )

    def _on_group_change(self, change) -> None:
        model_options = self._models_for_group(change["new"])
        selected = (model_options[0],) if model_options else ()
        self.controls["models"].options = model_options
        self.controls["models"].value = selected
        self.controls["models"].rows = min(max(len(model_options), 4), 12)
        self.controls["status"].value = self._status_html(f"{len(model_options)} models available", tone="muted")

    def _on_show_outliers(self, _) -> None:
        self.controls["view_mode"].value = "Outliers"
        self.render_current()

    def _summary_html(self) -> str:
        metrics = [
            ("Rows", f"{len(self.df):,}"),
            ("Datasets", f"{self.df['dataset'].nunique():,}"),
            ("Horizons", f"{self.df['horizon'].nunique():,}"),
            ("Groups", f"{self.df['model_group'].nunique():,}"),
        ]
        pills = "".join(
            "<span style='display:inline-flex;gap:6px;align-items:baseline;"
            "border:1px solid #d8dee9;background:#f8fafc;padding:5px 9px;margin-left:6px;"
            "font:12px/1.2 Inter,Segoe UI,Arial,sans-serif;color:#475569;'>"
            f"<b style='font-size:14px;color:#172033;'>{value}</b>{label}</span>"
            for label, value in metrics
        )
        return f"<div style='text-align:right;white-space:nowrap;'>{pills}</div>"

    def _status_html(self, message: str, *, tone: str) -> str:
        colors = {
            "success": ("#ecfdf5", "#047857", "#a7f3d0"),
            "error": ("#fef2f2", "#b91c1c", "#fecaca"),
            "muted": ("#f8fafc", "#475569", "#d8dee9"),
        }
        bg, fg, border = colors[tone]
        return (
            f"<span style='display:inline-block;border:1px solid {border};background:{bg};color:{fg};"
            "padding:7px 10px;font:12px/1.2 Inter,Segoe UI,Arial,sans-serif;'>"
            f"{message}</span>"
        )

    def _set_status(self, selection: DashboardSelection) -> None:
        model_count = len(selection.model_labels) or len(self._models_for_group(selection.model_group))
        self.controls["status"].value = self._status_html(
            f"Rendered {selection.view_mode} | {selection.dataset} h={selection.horizon} | {model_count} model(s)",
            tone="success",
        )

    def _styled_outlier_table(self, selection: DashboardSelection):
        table = self.outlier_table(selection)
        return table.style.format(
            {
                "true_volatility": "{:.6f}",
                "predict_volatility": "{:.6f}",
                "abs_error": "{:.6f}",
                "relative_error": "{:.4f}",
                "squared_error": "{:.6f}",
            }
        ).background_gradient(subset=["abs_error", "relative_error"], cmap="OrRd")

    def _models_for_group(self, model_group: str | None) -> list[str]:
        if model_group is None:
            return []
        data = self.df[self.df["model_group"].eq(model_group)]
        return sorted(data["model_label"].dropna().unique().tolist())

    def _filtered_rows(
        self,
        selection: DashboardSelection,
        *,
        model_labels: Iterable[str] | None = None,
        require_models: bool = True,
    ) -> pd.DataFrame:
        data = self.df[
            self.df["dataset"].eq(selection.dataset)
            & self.df["horizon"].eq(selection.horizon)
            & self.df["model_group"].eq(selection.model_group)
        ]
        labels = list(model_labels if model_labels is not None else selection.model_labels)
        if labels:
            data = data[data["model_label"].isin(labels)]
        elif require_models:
            raise ValueError("Select at least one model")
        return data.sort_values(["model_label", "time"]).copy()

    def _first_model(self, selection: DashboardSelection) -> str:
        if selection.model_labels:
            return selection.model_labels[0]
        options = self._models_for_group(selection.model_group)
        if not options:
            raise ValueError("No models are available for the selected group")
        return options[0]

    def _selected_or_all_group_models(self, selection: DashboardSelection) -> list[str]:
        return list(selection.model_labels) or self._models_for_group(selection.model_group)

    def _apply_common_layout(
        self,
        fig: go.Figure,
        *,
        title: str,
        y_title: str,
        log_y: bool,
        has_secondary_y: bool,
    ) -> None:
        fig.update_layout(
            title=title,
            template="plotly_white",
            height=620,
            hovermode="x unified",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
            margin={"l": 60, "r": 40, "t": 90, "b": 50},
        )
        fig.update_xaxes(title_text="Time", rangeslider={"visible": True})
        if has_secondary_y:
            fig.update_yaxes(title_text=y_title, type="log" if log_y else "linear", secondary_y=False)
        else:
            fig.update_yaxes(title_text=y_title, type="log" if log_y else "linear")
