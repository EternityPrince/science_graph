import sys
import os
from pathlib import Path
import subprocess
from typing import List, Dict, Any, Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False


def copy_to_clipboard(text: str) -> bool:
    """Copies the text to clipboard. Supports macOS pbcopy, Linux xclip/xsel, and Windows clip."""
    try:
        if sys.platform == "darwin":
            process = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            process.communicate(input=text.encode("utf-8"))
            return True
        else:
            if subprocess.run(["which", "xclip"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                process = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
                process.communicate(input=text.encode("utf-8"))
                return True
            elif subprocess.run(["which", "xsel"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                process = subprocess.Popen(["xsel", "--clipboard", "--input"], stdin=subprocess.PIPE)
                process.communicate(input=text.encode("utf-8"))
                return True
            elif os.name == "nt":
                process = subprocess.Popen(["clip"], stdin=subprocess.PIPE)
                process.communicate(input=text.encode("utf-8"))
                return True
    except Exception as e:
        print(f"Clipboard copy error: {e}")
    return False


def find_runs(reports_dir: Path) -> List[Path]:
    """Finds all run_* directories in reports_dir sorted by creation time (latest first)."""
    if not reports_dir.exists():
        print(f"Error: Reports directory {reports_dir} does not exist.")
        sys.exit(1)
        
    runs = [d for d in reports_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    runs.sort(key=lambda x: x.name, reverse=True)
    return runs


def get_metrics_summary(run_dir: Path, reports_dir: Path, script_dir: Path) -> str:
    """Reads or generates metrics_summary.md for the given run directory."""
    summary_md_path = run_dir / "metrics_summary.md"
    
    if summary_md_path.exists():
        with open(summary_md_path, "r", encoding="utf-8") as f:
            return f.read()
            
    # Auto-generate if missing but result_metrics.yaml exists
    result_metrics_path = run_dir / "result_metrics.yaml"
    if result_metrics_path.exists():
        parse_script = script_dir / "parse_metrics.py"
        if parse_script.exists():
            msg = f"[*] metrics_summary.md missing in {run_dir.name}. Auto-generating from result_metrics.yaml..."
            if HAS_RICH:
                console.print(f"[yellow]{msg}[/yellow]")
            else:
                print(msg)
            try:
                subprocess.run(
                    [sys.executable, str(parse_script), "--file", str(result_metrics_path), "--output-md", str(summary_md_path)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                if summary_md_path.exists():
                    with open(summary_md_path, "r", encoding="utf-8") as f:
                        return f.read()
            except Exception as e:
                print(f"Warning: Failed to auto-generate summary: {e}")

    # If neither exists, warn that this run has no evaluation metrics
    msg = f"[!] Warning: Run '{run_dir.name}' has not been evaluated yet (missing result_metrics.yaml)."
    if HAS_RICH:
        console.print(f"[bold red]{msg}[/bold red]")
        console.print(f"[dim]Falling back to reports/metrics_summary.md (global latest)[/dim]")
    else:
        print(msg)
        print("Falling back to reports/metrics_summary.md (global latest)")
        
    fallback_path = reports_dir / "metrics_summary.md"
    if fallback_path.exists():
        with open(fallback_path, "r", encoding="utf-8") as f:
            return f.read()
            
    print(f"Error: Could not find any metrics summary for run {run_dir.name}")
    sys.exit(1)
