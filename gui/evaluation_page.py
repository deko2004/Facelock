"""Evaluation page: FAR/FRR metrics, confusion matrix and threshold chart."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from config import settings
from core.evaluation import EvaluationReport
from gui.styles import COLORS, font
from gui.widgets import BasePage, Card, PageHeader, StatCard


class EvaluationPage(BasePage):
    def __init__(self, parent: tk.Misc, controller) -> None:
        super().__init__(parent, controller)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)
        self._last_report: EvaluationReport | None = None
        self._chart_job: str | None = None

        PageHeader(
            self,
            "Evaluation & Metrics",
            "Compute genuine and impostor comparisons on a folder of identity "
            "subfolders. Metrics are only meaningful when the dataset is adequate.",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 14))

        controls = Card(self, padding=12)
        controls.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Dataset folder", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        default_dataset = settings.IMAGES_DIR / "eval_demo"
        self.dataset_var = tk.StringVar(value=str(default_dataset))
        ttk.Entry(controls, textvariable=self.dataset_var).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Button(
            controls, text="Browse...", style="Secondary.TButton", command=self._browse
        ).grid(row=0, column=2, padx=8)
        ttk.Label(controls, text="Threshold", style="Card.TLabel").grid(
            row=0, column=3, sticky="w", padx=(8, 6)
        )
        self.threshold_var = tk.StringVar(value=f"{self.ctx.threshold:.3f}")
        ttk.Entry(controls, textvariable=self.threshold_var, width=8).grid(
            row=0, column=4
        )
        self.run_button = ttk.Button(
            controls, text="Run evaluation", style="Primary.TButton", command=self._run
        )
        self.run_button.grid(row=0, column=5, padx=(12, 0))
        ttk.Label(
            controls,
            text=(
                "Expected layout: dataset/<identity>/<images>. The included "
                "images/eval_demo set is a small public smoke-test sample, not a "
                "biometric benchmark."
            ),
            style="CardMuted.TLabel",
            wraplength=1100,
        ).grid(row=1, column=0, columnspan=6, sticky="w", pady=(8, 0))

        stats = ttk.Frame(self, style="App.TFrame")
        stats.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        for index in range(6):
            stats.grid_columnconfigure(index, weight=1, uniform="evalstats")
        self.comparisons_stat = StatCard(stats, "Comparisons", "0", "All evaluated pairs")
        self.comparisons_stat.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.pairs_stat = StatCard(stats, "Genuine / impostor", "0 / 0", "Pair counts")
        self.pairs_stat.grid(row=0, column=1, sticky="nsew", padx=6)
        self.far_stat = StatCard(stats, "FAR", "N/A", "False Acceptance Rate")
        self.far_stat.grid(row=0, column=2, sticky="nsew", padx=6)
        self.frr_stat = StatCard(stats, "FRR", "N/A", "False Rejection Rate")
        self.frr_stat.grid(row=0, column=3, sticky="nsew", padx=6)
        self.accuracy_stat = StatCard(stats, "Accuracy", "N/A", "(TP + TN) / total")
        self.accuracy_stat.grid(row=0, column=4, sticky="nsew", padx=6)
        self.precision_stat = StatCard(
            stats, "Precision / recall", "N/A", "Where defined by the sample"
        )
        self.precision_stat.grid(row=0, column=5, sticky="nsew", padx=(6, 0))

        body = ttk.Frame(self, style="App.TFrame")
        body.grid(row=3, column=0, sticky="nsew")
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        chart_card = Card(body)
        chart_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        chart_card.columnconfigure(0, weight=1)
        chart_card.rowconfigure(1, weight=1)
        ttk.Label(
            chart_card, text="FAR / FRR trade-off", style="CardTitle.TLabel"
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.chart_canvas = tk.Canvas(
            chart_card,
            bg=COLORS["panel"],
            highlightthickness=0,
            height=330,
        )
        self.chart_canvas.grid(row=1, column=0, sticky="nsew")
        self.chart_canvas.bind("<Configure>", self._chart_resized)

        side = ttk.Frame(body, style="App.TFrame")
        side.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        side.columnconfigure(0, weight=1)
        side.rowconfigure(1, weight=1)

        matrix_card = Card(side)
        matrix_card.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(
            matrix_card, text="Confusion matrix", style="CardTitle.TLabel"
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        self.matrix_vars = {
            "tp": tk.StringVar(value="-"),
            "fn": tk.StringVar(value="-"),
            "fp": tk.StringVar(value="-"),
            "tn": tk.StringVar(value="-"),
        }
        matrix_items = [
            ("True accept (TP)", "tp", 1, 0),
            ("False reject (FN)", "fn", 1, 1),
            ("False accept (FP)", "fp", 2, 0),
            ("True reject (TN)", "tn", 2, 1),
        ]
        for title, key, row, column in matrix_items:
            frame = ttk.Frame(matrix_card, style="CardAlt.TFrame", padding=8)
            frame.grid(row=row, column=column, sticky="nsew", padx=3, pady=3)
            ttk.Label(frame, text=title, style="CardMuted.TLabel").pack(anchor="w")
            tk.Label(
                frame,
                textvariable=self.matrix_vars[key],
                bg=COLORS["card_alt"],
                fg=COLORS["text"],
                font=font(18, "bold"),
            ).pack(anchor="w")
        for column in range(2):
            matrix_card.grid_columnconfigure(column, weight=1)

        info_card = Card(side)
        info_card.grid(row=1, column=0, sticky="nsew")
        info_card.columnconfigure(0, weight=1)
        info_card.rowconfigure(1, weight=1)
        ttk.Label(info_card, text="Notes and warnings", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        self.info_text = tk.Text(
            info_card,
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            insertbackground=COLORS["text"],
            font=font(9),
            wrap="word",
            borderwidth=0,
            highlightthickness=0,
            height=8,
        )
        self.info_text.grid(row=1, column=0, sticky="nsew")
        self.info_text.configure(state="disabled")
        self.status_label = ttk.Label(
            info_card, text="No evaluation has been run.", style="CardMuted.TLabel"
        )
        self.status_label.grid(row=2, column=0, sticky="w", pady=(6, 0))

        self._draw_empty_chart()

    def on_show(self) -> None:
        self.threshold_var.set(f"{self.ctx.threshold:.3f}")

    def _browse(self) -> None:
        folder = filedialog.askdirectory(title="Choose an evaluation dataset folder")
        if folder:
            self.dataset_var.set(folder)

    def _threshold(self) -> float | None:
        try:
            value = float(self.threshold_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid threshold", "Enter a number between 0 and 1.")
            return None
        if value < 0.0 or value > 1.0:
            messagebox.showerror("Invalid threshold", "The threshold must be between 0 and 1.")
            return None
        return value

    def _run(self) -> None:
        dataset = self.dataset_var.get().strip()
        if not dataset:
            messagebox.showwarning("No dataset", "Choose a dataset folder first.")
            return
        threshold = self._threshold()
        if threshold is None:
            return
        self.run_button.configure(state="disabled")
        self.status_label.configure(text="Evaluating...", foreground=COLORS["muted"])
        self._set_info("Running YuNet + SFace comparisons. Please wait...")

        def work():
            report = self.ctx.evaluation.evaluate(dataset, threshold=threshold)
            output_dir = settings.EVALUATION_DIR
            pairs_path, summary_path = self.ctx.evaluation.write_csv(
                report,
                output_dir / "evaluation_pairs.csv",
                output_dir / "evaluation_summary.csv",
            )
            chart_path = self.ctx.evaluation.render_chart(
                report, output_dir / "evaluation_chart.png"
            )
            return report, pairs_path, summary_path, chart_path

        self.run_background(work, self._show_report, self._run_failed)

    def _run_failed(self, exc: Exception) -> None:
        self.run_button.configure(state="normal")
        self.status_label.configure(text=f"Evaluation failed: {exc}", foreground=COLORS["danger"])
        self._set_info(str(exc))
        messagebox.showerror("Evaluation failed", str(exc))

    def _show_report(self, result) -> None:
        report, pairs_path, summary_path, chart_path = result
        self._last_report = report
        self.run_button.configure(state="normal")
        metrics = report.metrics
        self.comparisons_stat.set_value(str(len(report.pairs)))
        self.pairs_stat.set_value(
            f"{report.genuine_count} / {report.impostor_count}"
        )
        self.far_stat.set_value(self._percent(metrics.far))
        self.frr_stat.set_value(self._percent(metrics.frr))
        self.accuracy_stat.set_value(self._percent(metrics.accuracy))
        precision = self._percent(metrics.precision)
        recall = self._percent(metrics.recall)
        self.precision_stat.set_value(f"{precision} / {recall}")
        self.matrix_vars["tp"].set(str(metrics.tp))
        self.matrix_vars["fn"].set(str(metrics.fn))
        self.matrix_vars["fp"].set(str(metrics.fp))
        self.matrix_vars["tn"].set(str(metrics.tn))

        lines = [
            f"Dataset: {report.dataset_dir}",
            f"Identities: {len(report.identities)}  |  usable images: {report.image_count}",
            f"Threshold: {report.threshold:.3f}  |  elapsed: {report.elapsed_seconds:.2f}s",
            f"CSV: {pairs_path}",
        ]
        if summary_path is not None:
            lines.append(f"Summary CSV: {summary_path}")
        lines.append(f"Chart: {chart_path}")
        if report.warnings:
            lines.append("")
            lines.extend(f"Warning: {warning}" for warning in report.warnings)
        if report.skipped:
            lines.append("")
            lines.append(f"Skipped {len(report.skipped)} image(s):")
            lines.extend(f"  {item}" for item in report.skipped[:8])
        self._set_info("\n".join(lines))
        self.status_label.configure(
            text="Evaluation complete. CSV and chart saved.", foreground=COLORS["success"]
        )
        self._draw_chart(report)

    def _set_info(self, text: str) -> None:
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", text)
        self.info_text.configure(state="disabled")

    @staticmethod
    def _percent(value: float | None) -> str:
        return "N/A" if value is None else f"{value * 100:.1f}%"

    def _chart_resized(self, _event=None) -> None:
        if self._chart_job is not None:
            try:
                self.after_cancel(self._chart_job)
            except tk.TclError:
                pass
        self._chart_job = self.after(120, self._redraw_current_chart)

    def _redraw_current_chart(self) -> None:
        self._chart_job = None
        if self._last_report is not None:
            self._draw_chart(self._last_report)
        else:
            self._draw_empty_chart()

    def _draw_empty_chart(self) -> None:
        canvas = self.chart_canvas
        canvas.delete("all")
        width = max(300, canvas.winfo_width())
        height = max(200, canvas.winfo_height())
        canvas.create_text(
            width / 2,
            height / 2,
            text="Run an evaluation to plot FAR and FRR against the threshold.",
            fill=COLORS["muted"],
            font=font(10),
        )

    def _draw_chart(self, report: EvaluationReport) -> None:
        canvas = self.chart_canvas
        canvas.delete("all")
        width = max(420, canvas.winfo_width())
        height = max(240, canvas.winfo_height())
        left, right, top, bottom = 56, 24, 24, 42
        plot_w = width - left - right
        plot_h = height - top - bottom

        def x_for(value: float) -> float:
            return left + value * plot_w

        def y_for(value: float) -> float:
            return top + (1.0 - value) * plot_h

        for step in range(0, 11):
            value = step / 10.0
            x = x_for(value)
            y = y_for(value)
            canvas.create_line(x, top, x, top + plot_h, fill=COLORS["border"])
            canvas.create_line(left, y, left + plot_w, y, fill=COLORS["border"])
            canvas.create_text(x, top + plot_h + 14, text=f"{value:.1f}", fill=COLORS["muted"], font=font(8))
            canvas.create_text(left - 20, y, text=f"{value:.1f}", fill=COLORS["muted"], font=font(8))
        canvas.create_text(left + plot_w / 2, height - 8, text="Cosine threshold", fill=COLORS["muted"], font=font(8))
        canvas.create_text(14, top + plot_h / 2, text="Rate", fill=COLORS["muted"], font=font(8), angle=90)

        def series(values: list[float | None], colour: str, label: str, legend_row: int) -> None:
            points: list[float] = []
            for index, value in enumerate(values):
                if value is None:
                    continue
                points.extend([x_for(index / 100.0), y_for(value)])
            if len(points) >= 4:
                canvas.create_line(*points, fill=colour, width=3, smooth=False)
            canvas.create_line(
                left + plot_w - 110,
                top + 10 + legend_row * 18,
                left + plot_w - 92,
                top + 10 + legend_row * 18,
                fill=colour,
                width=3,
            )
            canvas.create_text(
                left + plot_w - 86,
                top + 10 + legend_row * 18,
                text=label,
                fill=COLORS["text"],
                anchor="w",
                font=font(8),
            )

        series([metric.far for metric in report.sweep], COLORS["danger"], "FAR", 0)
        series([metric.frr for metric in report.sweep], "#4cc9f0", "FRR", 1)
        threshold_x = x_for(report.threshold)
        canvas.create_line(
            threshold_x,
            top,
            threshold_x,
            top + plot_h,
            fill=COLORS["warning"],
            width=2,
            dash=(4, 3),
        )
        canvas.create_text(
            min(threshold_x + 6, width - 90),
            top + plot_h - 12,
            text=f"selected {report.threshold:.3f}",
            fill=COLORS["warning"],
            anchor="w",
            font=font(8),
        )
        canvas.create_text(
            left,
            height - 8,
            text=f"genuine={report.genuine_count}  impostor={report.impostor_count}",
            fill=COLORS["muted"],
            anchor="w",
            font=font(8),
        )
