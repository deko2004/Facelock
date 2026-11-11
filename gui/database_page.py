"""User database page: list, inspect and delete encrypted templates."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from core.database import DatabaseError, UserSummary
from gui.styles import COLORS
from gui.widgets import BasePage, Card, PageHeader


class DatabasePage(BasePage):
    def __init__(self, parent: tk.Misc, controller) -> None:
        super().__init__(parent, controller)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=0)
        self._summaries: dict[str, UserSummary] = {}

        PageHeader(
            self,
            "User Database",
            "Only encrypted templates and non-biometric metadata are stored. "
            "Feature vectors are never displayed or written to logs.",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 14))

        toolbar = Card(self, padding=12)
        toolbar.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        toolbar.columnconfigure(2, weight=1)
        ttk.Button(
            toolbar, text="Refresh", style="Secondary.TButton", command=self._refresh
        ).grid(row=0, column=0, padx=(0, 8))
        self.delete_button = ttk.Button(
            toolbar,
            text="Delete selected user",
            style="Danger.TButton",
            command=self._delete_selected,
            state="disabled",
        )
        self.delete_button.grid(row=0, column=1)
        self.count_label = ttk.Label(
            toolbar, text="0 enrolled users", style="CardMuted.TLabel"
        )
        self.count_label.grid(row=0, column=2, sticky="e")

        table_card = Card(self, padding=12)
        table_card.grid(row=2, column=0, sticky="nsew", pady=(0, 12))
        table_card.columnconfigure(0, weight=1)
        table_card.rowconfigure(1, weight=1)
        ttk.Label(table_card, text="Enrolled users", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        columns = ("user_id", "display_name", "created", "template", "dim", "model")
        self.tree = ttk.Treeview(
            table_card, columns=columns, show="headings", height=9
        )
        headings = {
            "user_id": ("User ID", 130),
            "display_name": ("Display name", 190),
            "created": ("Created (UTC)", 175),
            "template": ("Encrypted template", 145),
            "dim": ("Embedding", 95),
            "model": ("Model", 180),
        }
        for column, (title, width) in headings.items():
            self.tree.heading(column, text=title)
            self.tree.column(column, width=width, anchor="w")
        self.tree.grid(row=1, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", self._selection_changed)
        scrollbar = ttk.Scrollbar(
            table_card, orient="vertical", command=self.tree.yview
        )
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)

        details = Card(self, padding=12)
        details.grid(row=3, column=0, sticky="ew")
        details.columnconfigure(1, weight=1)
        details.columnconfigure(3, weight=1)
        ttk.Label(details, text="Selected user", style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 8)
        )
        self.detail_vars = {
            "user_id": tk.StringVar(value="-"),
            "display_name": tk.StringVar(value="-"),
            "created": tk.StringVar(value="-"),
            "template": tk.StringVar(value="-"),
            "model": tk.StringVar(value="-"),
            "dimension": tk.StringVar(value="-"),
        }
        fields = [
            ("User ID", "user_id", 1, 0),
            ("Display name", "display_name", 3, 0),
            ("Created", "created", 1, 1),
            ("Encrypted template", "template", 3, 1),
            ("Model", "model", 1, 2),
            ("Embedding dimension", "dimension", 3, 2),
        ]
        for title, key, column, row in fields:
            ttk.Label(details, text=title, style="CardMuted.TLabel").grid(
                row=row * 2 + 1, column=column - 1, sticky="w", padx=(0, 10), pady=(4, 0)
            )
            ttk.Label(
                details, textvariable=self.detail_vars[key], style="Card.TLabel"
            ).grid(
                row=row * 2 + 2,
                column=column - 1,
                columnspan=2,
                sticky="w",
                pady=(0, 4),
            )

        ttk.Label(
            details,
            text=(
                "Deleting a user removes the encrypted template from the database "
                "and records a deletion event in the audit log. The original "
                "photograph was never stored."
            ),
            style="CardMuted.TLabel",
            wraplength=1000,
        ).grid(row=7, column=0, columnspan=4, sticky="w", pady=(8, 0))

    def on_show(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        try:
            users = self.ctx.database.list_users()
        except DatabaseError as exc:
            messagebox.showerror("Database error", str(exc))
            return
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._summaries = {}
        for user in users:
            item = self.tree.insert(
                "",
                "end",
                values=(
                    user.user_id,
                    user.display_name,
                    user.created_at.replace("T", " ")[:19],
                    f"{user.template_size} bytes",
                    f"{user.embedding_dim}-D",
                    user.model,
                ),
            )
            self._summaries[item] = user
        self.count_label.configure(
            text=f"{len(users)} enrolled user{'s' if len(users) != 1 else ''}"
        )
        self.delete_button.configure(state="disabled")
        self._clear_details()
        children = self.tree.get_children()
        if children:
            self.tree.selection_set(children[0])
            self.tree.focus(children[0])
            self._selection_changed()

    def _clear_details(self) -> None:
        for var in self.detail_vars.values():
            var.set("-")

    def _selection_changed(self, _event=None) -> None:
        selection = self.tree.selection()
        if not selection:
            self.delete_button.configure(state="disabled")
            self._clear_details()
            return
        user = self._summaries.get(selection[0])
        if user is None:
            return
        self.delete_button.configure(state="normal")
        self.detail_vars["user_id"].set(user.user_id)
        self.detail_vars["display_name"].set(user.display_name)
        self.detail_vars["created"].set(user.created_at.replace("T", " ")[:19])
        self.detail_vars["template"].set(f"{user.template_size} bytes (encrypted)")
        self.detail_vars["model"].set(user.model)
        self.detail_vars["dimension"].set(f"{user.embedding_dim} float32 values")

    def _delete_selected(self) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        user = self._summaries.get(selection[0])
        if user is None:
            return
        confirmed = messagebox.askyesno(
            "Delete user",
            f"Delete '{user.display_name}' ({user.user_id}) and the encrypted "
            "template?\n\nThis cannot be undone.",
            icon="warning",
        )
        if not confirmed:
            return
        try:
            deleted = self.ctx.database.delete_user(user.user_id)
        except Exception as exc:
            messagebox.showerror("Delete failed", str(exc))
            return
        if deleted:
            messagebox.showinfo("Deleted", f"User '{user.user_id}' was deleted.")
        else:
            messagebox.showwarning("Not found", "The user no longer exists.")
        self._refresh()
