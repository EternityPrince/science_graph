import os
import sys
import time
import subprocess
from rich.progress import ProgressColumn
from rich.text import Text
from src import console as con

class IterationSpeedColumn(ProgressColumn):
    def render(self, task):
        speed = task.finished_speed or task.speed
        if speed is None or speed == 0:
            return Text("- sec/it", style="progress.data.speed")
        sec_per_it = 1.0 / speed
        return Text(f"{sec_per_it:.2f} sec/it", style="progress.data.speed")


def run_command_with_progress(cmd: list, title: str, total_steps: int, step_pattern: str) -> float:
    from rich.progress import (
        Progress,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        MofNCompleteColumn
    )
    t0 = time.perf_counter()
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, finished_style="green"),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        IterationSpeedColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=con.console
    ) as progress:
        task = progress.add_task(f"[cyan]{title}", total=total_steps)
        
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        
        for line in iter(process.stdout.readline, ""):
            line_str = line.strip()
            line_lower = line_str.lower()
            
            # Parse progress based on patterns
            if step_pattern == "retrieval":
                if "Query: '" in line_str or "] Query:" in line_str:
                    progress.advance(task, 1)
            elif step_pattern == "generation":
                if line_str.startswith("Running ") and ":" in line_str and not line_str.startswith("Running command:"):
                    progress.advance(task, 1)
            elif step_pattern == "evaluation":
                if "Evaluated case " in line_str:
                    try:
                        parts = line_str.split("Evaluated case ")[1].split("/")
                        current = int(parts[0])
                        progress.update(task, completed=current)
                    except Exception:
                        pass
            
            # Print warnings, errors, query/baseline execution updates or success milestones directly above the progress bar
            if "✗" in line_str or "error" in line_lower or "warning" in line_lower or "failed" in line_lower:
                progress.console.print(f"  [red]✗[/] {line_str}")
            elif "Loaded" in line_str or "Successfully" in line_str:
                progress.console.print(f"  [green]✓[/] {line_str}")
            elif "Ready" in line_str.lower() or "initializing" in line_str.lower():
                progress.console.print(f"  [dim]{line_str}[/]")
            elif (
                "query" in line_lower or
                "running b" in line_lower or
                "completed" in line_lower or
                "generating answer" in line_lower or
                "evaluated" in line_lower or
                "evaluation finished" in line_lower or
                "benchmarking complete" in line_lower or
                line_str.startswith("===") or line_str.endswith("===")
            ):
                progress.console.print(f"  {line_str}")
                
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
            
    return time.perf_counter() - t0
