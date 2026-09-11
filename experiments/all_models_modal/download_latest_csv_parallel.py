"""Discover the newest Modal Volume run containing CSV files and download them.

 test:  py -3.11 experiments\all_models_modal\download_latest_csv_parallel.py --dry-run
 run full: py -3.11 experiments\all_models_modal\download_latest_csv_parallel.py
Windows-friendly multiprocessing downloader.  Every worker creates its own
Modal Volume handle; Modal clients are not shared between processes.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

import modal
from modal.types import FileEntryType
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from rich.text import Text


RUN_RE = re.compile(r"\d{8}_\d{6}")
_worker_volume = None
console = Console()


def _init_worker(volume_name: str) -> None:
    """Create one independent Modal client/Volume handle in each process."""
    global _worker_volume
    _worker_volume = modal.Volume.from_name(volume_name)


def _download_one(job: tuple[str, str, int]) -> tuple[str, str, int]:
    """Download one file atomically, or skip it when already complete."""
    remote_path, local_path, expected_size = job
    destination = Path(local_path)
    partial = Path(str(destination) + ".part")

    if destination.exists() and destination.stat().st_size == expected_size:
        return remote_path, "skipped", expected_size

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial.unlink(missing_ok=True)

    try:
        with partial.open("wb") as fileobj:
            downloaded_size = _worker_volume.read_file_into_fileobj(
                remote_path, fileobj
            )

        actual_size = partial.stat().st_size
        if downloaded_size != expected_size or actual_size != expected_size:
            raise IOError(
                f"size mismatch: expected {expected_size}, "
                f"downloaded={downloaded_size}, local={actual_size}"
            )

        os.replace(partial, destination)
        return remote_path, "downloaded", expected_size
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _discover(volume_name: str) -> tuple[str, list[tuple[str, str, int]], list]:
    """Scan the complete Volume tree and select the newest run with CSVs."""
    volume = modal.Volume.from_name(volume_name)
    entries = volume.listdir("/", recursive=True)
    files_by_run: dict[str, list] = {}

    for entry in entries:
        if entry.type != FileEntryType.FILE:
            continue
        if not entry.path.lower().endswith(".csv"):
            continue

        parts = PurePosixPath(entry.path).parts
        if not parts or not RUN_RE.fullmatch(parts[0]):
            continue
        files_by_run.setdefault(parts[0], []).append(entry)

    if not files_by_run:
        raise FileNotFoundError("No timestamped run containing CSV files was found")

    run = max(files_by_run)
    jobs = []
    for entry in sorted(files_by_run[run], key=lambda item: item.path):
        relative = PurePosixPath(entry.path).relative_to(run)
        local_path = Path("output/final") / run / Path(*relative.parts)
        jobs.append((entry.path, str(local_path), entry.size))

    return run, jobs, entries


def _format_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(value)
    for unit in units:
        if abs(value) < 1024 or unit == units[-1]:
            return f"{value:,.1f} {unit}"
        value /= 1024
    return f"{value:,.1f} TiB"


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "--:--:--"
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _short_path(path: str, width: int = 68) -> str:
    if len(path) <= width:
        return path
    return "…" + path[-(width - 1):]


def _dashboard(
    *,
    volume_name: str,
    run: str,
    total_files: int,
    completed_files: int,
    failed_files: int,
    completed_bytes: int,
    total_bytes: int,
    workers: int,
    active_jobs: list[tuple[str, str]],
    queued_jobs: int,
    failures: list[tuple[str, str]],
    elapsed: float,
    progress: Progress,
) -> Group:
    rate = completed_bytes / elapsed if elapsed > 0 else 0.0
    remaining = total_bytes - completed_bytes
    eta = remaining / rate if rate > 0 else None
    percent = (completed_bytes / total_bytes * 100) if total_bytes else 100.0

    title = Text(" MODAL VOLUME CSV DOWNLOADER ", style="bold white on blue")
    subtitle = Text(f"  {volume_name}  •  run {run}", style="bright_cyan")
    header = Table.grid(expand=True, padding=(0, 1))
    header.add_column(ratio=2)
    header.add_column(ratio=1)
    header.add_column(ratio=1)
    header.add_column(ratio=1)
    header.add_row(
        Panel(Group(title, subtitle), border_style="bright_blue"),
        Panel(
            Text.assemble(("FILES\n", "dim"), (f"{completed_files:,}/{total_files:,}", "bold green")),
            title="COMPLETED",
            border_style="green",
        ),
        Panel(
            Text.assemble(("WORKERS\n", "dim"), (f"{workers:,}", "bold yellow"), (f"\nACTIVE {len(active_jobs):,}", "dim")),
            title="PARALLEL",
            border_style="yellow",
        ),
        Panel(
            Text.assemble(("FAILED\n", "dim"), (f"{failed_files:,}", "bold red")),
            title="ERRORS",
            border_style="red" if failed_files else "green",
        ),
    )

    stats = Table.grid(expand=True, padding=(0, 2))
    stats.add_column(justify="left")
    stats.add_column(justify="left")
    stats.add_column(justify="left")
    stats.add_column(justify="left")
    stats.add_row(
        f"[bold cyan]PROGRESS[/] {percent:6.2f}%",
        f"[bold cyan]DATA[/] {_format_bytes(completed_bytes)} / {_format_bytes(total_bytes)}",
        f"[bold cyan]SPEED[/] {_format_bytes(rate)}/s",
        f"[bold cyan]ETA[/] {_format_duration(eta)}",
    )

    active = Table(expand=True, box=box.SIMPLE_HEAD, header_style="bold bright_cyan")
    active.add_column("STATE", width=10)
    active.add_column("REMOTE CSV", ratio=1)
    active.add_column("PROCESS", justify="right", width=9)
    if active_jobs:
        for index, (remote_path, state) in enumerate(active_jobs[:14], start=1):
            active.add_row(
                f"[yellow]{state}[/]",
                f"[white]{_short_path(remote_path)}[/]",
                f"[dim]#{index:02d}[/]",
            )
        if len(active_jobs) > 14:
            active.add_row("[dim]…[/]", f"[dim]+ {len(active_jobs) - 14:,} more active[/]", "")
    else:
        active.add_row("[green]IDLE[/]", "Waiting for workers…", "")

    queue_text = f"[bold yellow]{queued_jobs:,}[/] queued" if queued_jobs else "[green]queue empty[/]"
    active_panel = Panel(active, title=f"ACTIVE TRANSFERS  •  {queue_text}", border_style="bright_cyan")

    error_panel = None
    if failures:
        errors = Table.grid(expand=True)
        for remote_path, error in failures[-3:]:
            errors.add_row(f"[red]✖[/] {_short_path(remote_path, 58)}  [dim]{_short_path(error, 52)}[/]")
        error_panel = Panel(errors, title="RECENT ERRORS", border_style="red")

    body = [Panel(header, border_style="bright_blue"), Panel(stats, border_style="blue"), progress, active_panel]
    if error_panel:
        body.append(error_panel)
    return Group(*body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", default="Results")
    parser.add_argument("--output-dir", default="output/final")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="processes; 0 = all available CPUs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="scan and print the newest CSV run without downloading",
    )
    args = parser.parse_args()

    run, discovered_jobs, entries = _discover(args.volume)
    jobs = [
        (remote, str(Path(args.output_dir) / run / Path(*PurePosixPath(remote).relative_to(run).parts)), size)
        for remote, _, size in discovered_jobs
    ]

    top_level = Counter(
        PurePosixPath(remote).parts[1] if len(PurePosixPath(remote).parts) > 1 else "."
        for remote, _, _ in jobs
    )
    total_bytes = sum(size for _, _, size in jobs)
    console.print(
        Panel(
            f"[bold cyan]Volume[/] {args.volume}\n"
            f"[bold cyan]Scanned[/] {len(entries):,} entries\n"
            f"[bold cyan]Newest run with CSV[/] {run}\n"
            f"[bold cyan]CSV files[/] {len(jobs):,}  [bold cyan]Size[/] {total_bytes / 1024**2:,.1f} MiB\n"
            f"[bold cyan]Folders[/] {', '.join(f'{name}={count:,}' for name, count in sorted(top_level.items()))}",
            title="[bold white]DISCOVERY COMPLETE[/]",
            border_style="bright_blue",
        )
    )

    if args.dry_run:
        return 0

    destination = Path(args.output_dir) / run
    destination.mkdir(parents=True, exist_ok=True)

    pending = []
    already_complete = 0
    completed_bytes = 0
    for job in jobs:
        if Path(job[1]).exists() and Path(job[1]).stat().st_size == job[2]:
            already_complete += 1
            completed_bytes += job[2]
        else:
            pending.append(job)

    cpu_count = os.cpu_count() or 1
    worker_count = min(len(pending), args.workers or cpu_count) if pending else 0
    failures: list[tuple[str, str]] = []
    started_at = time.monotonic()

    progress = Progress(
        TextColumn("[bold cyan]TOTAL[/]"),
        BarColumn(bar_width=None, complete_style="bright_green", finished_style="green"),
        TaskProgressColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        expand=True,
    )
    progress_task = progress.add_task("download", total=total_bytes, completed=completed_bytes)

    with Live(
        _dashboard(
            volume_name=args.volume,
            run=run,
            total_files=len(jobs),
            completed_files=already_complete,
            failed_files=0,
            completed_bytes=completed_bytes,
            total_bytes=total_bytes,
            workers=worker_count,
            active_jobs=[],
            queued_jobs=len(pending),
            failures=[],
            elapsed=0,
            progress=progress,
        ),
        console=console,
        refresh_per_second=5,
        screen=True,
    ) as live:
        if pending:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
                initializer=_init_worker,
                initargs=(args.volume,),
            ) as pool:
                future_to_job = {pool.submit(_download_one, job): job for job in pending}
                remaining_futures = set(future_to_job)

                while remaining_futures:
                    done = {future for future in remaining_futures if future.done()}
                    if not done:
                        time.sleep(0.2)
                    for future in done:
                        job = future_to_job[future]
                        remaining_futures.remove(future)
                        try:
                            _, _, size = future.result()
                            completed_files = already_complete + (len(pending) - len(remaining_futures))
                            completed_bytes += size
                            progress.update(progress_task, completed=completed_bytes)
                        except Exception as exc:
                            failures.append((job[0], repr(exc)))

                    active_jobs = [
                        (future_to_job[future][0], "RUNNING")
                        for future in remaining_futures
                        if future.running()
                    ]
                    completed_files = already_complete + (len(pending) - len(remaining_futures) - len(failures))
                    live.update(
                        _dashboard(
                            volume_name=args.volume,
                            run=run,
                            total_files=len(jobs),
                            completed_files=completed_files,
                            failed_files=len(failures),
                            completed_bytes=completed_bytes,
                            total_bytes=total_bytes,
                            workers=worker_count,
                            active_jobs=active_jobs,
                            queued_jobs=max(0, len(remaining_futures) - len(active_jobs)),
                            failures=failures,
                            elapsed=time.monotonic() - started_at,
                            progress=progress,
                        )
                    )

    if failures:
        console.print(Panel(f"[bold red]{len(failures)} file(s) failed.[/] See RECENT ERRORS in the dashboard.", border_style="red"))
        return 1

    console.print(
        Panel(
            f"[bold green]DOWNLOAD COMPLETE[/]\n"
            f"Downloaded: {len(jobs) - already_complete:,}\n"
            f"Already complete: {already_complete:,}\n"
            f"Destination: {destination}",
            border_style="bright_green",
        )
    )
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
