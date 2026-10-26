"""Обвязка скриптов экспериментов: команды ``run`` и ``plot``.

    python scripts/e1_precision.py run [--quick] [--jobs 8]
    python scripts/e1_precision.py plot runs/<...>_e1

``run`` считает, сохраняет папку запуска и строит рисунки из неё (без
``--no-plot``). ``plot`` ничего не считает: читает только папку запуска.
Рисунки и таблицы ``*_table.csv`` идут в ``figures/``; у быстрого прогона
(``--quick``) к именам добавляется ``_quick``, чтобы не затереть рисунки
статьи. Проверки печатаются как ``[OK]`` / ``[FAIL]``; если хоть одна не
прошла, код возврата 1.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from photosim.experiment import RUNS_DIR, Run, load_run
from photosim.figures import style

__all__ = ["FIGURES_DIR", "Output", "main"]

#: Куда по умолчанию идут рисунки и таблицы.
FIGURES_DIR = Path("figures")


class Output:
    """Рисунки, таблицы и проверки одного запуска."""

    def __init__(self, run: Run, out_dir: str | Path) -> None:
        self.dir = Path(out_dir)
        self.suffix = "_quick" if run.configs.get("quick") else ""
        self.failed: list[str] = []

    def figure(self, fig: Figure, name: str) -> None:
        for path in style.save(fig, name + self.suffix, self.dir):
            print(f"рисунок: {path}")
        plt.close(fig)

    def table(self, df: pd.DataFrame, name: str, header: str | None = None) -> Path:
        """``header`` — строки ``# ...`` перед таблицей (формула модели и т. п.);
        читать такую таблицу — ``pd.read_csv(path, comment="#")``."""
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{name}{self.suffix}_table.csv"
        with path.open("w") as f:
            for line in (header or "").splitlines():
                f.write(f"# {line}\n")
            df.to_csv(f, index=False, float_format="%.6g")
        print(f"таблица: {path}")
        return path

    def check(self, ok: bool, message: str) -> bool:
        print(f"[{'OK' if ok else 'FAIL'}] {message}")
        if not ok:
            self.failed.append(message)
        return bool(ok)


def main(
    name: str,
    run: Callable[[argparse.Namespace], Path],
    plot: Callable[[Run, Output, argparse.Namespace], None],
    *,
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
    argv: list[str] | None = None,
) -> int:
    """Разобрать команду и выполнить ``run`` или ``plot``; вернуть код выхода.

    ``run(args)`` считает и возвращает путь сохранённой папки запуска.
    ``plot(run, out, args)`` строит рисунки и таблицы только из ``run``.
    ``add_arguments`` добавляет параметры обеим командам.
    """
    parser = argparse.ArgumentParser(prog=Path(sys.argv[0]).name, description=name)
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="посчитать и сохранить папку запуска")
    p_run.add_argument(
        "--quick", action="store_true", help="грубая сетка и мало реализаций"
    )
    p_run.add_argument("--jobs", type=int, default=1, help="процессов (-1 — все)")
    p_run.add_argument("--root", type=Path, default=RUNS_DIR, help="папка запусков")
    p_run.add_argument("--no-plot", action="store_true", help="не строить рисунки")
    p_plot = sub.add_parser("plot", help="рисунки и таблицы из папки запуска")
    p_plot.add_argument("run_dir", type=Path)
    for p in (p_run, p_plot):
        p.add_argument("--out", type=Path, default=FIGURES_DIR, help="папка рисунков")
        if add_arguments is not None:
            add_arguments(p)
    args = parser.parse_args(argv)

    if args.command == "run":
        run_dir = run(args)
        print(f"запуск сохранён: {run_dir}")
        if args.no_plot:
            return 0
    else:
        run_dir = args.run_dir
    style.apply()
    out = Output(loaded := load_run(run_dir), args.out)
    plot(loaded, out, args)
    if out.failed:
        print(f"не прошло проверок: {len(out.failed)}")
        return 1
    return 0
