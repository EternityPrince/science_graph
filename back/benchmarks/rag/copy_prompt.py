#!/usr/bin/env python3
"""
Science Graph — Copy Benchmark Run Results Prompt to Clipboard.
Finds benchmarking runs in reports/, formats the metrics summary,
injects it into a prompt template, and copies the result to the clipboard.
"""

import sys
import os
import argparse
import subprocess
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional

# Try importing rich for prettier CLI
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, IntPrompt
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False

SCRIPT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = SCRIPT_DIR / "reports"
PROMPTS_FILE = SCRIPT_DIR / "prompts" / "analysis_prompts.yaml"


def load_templates() -> Dict[str, Any]:
    """Loads prompt templates from the yaml file."""
    if not PROMPTS_FILE.exists():
        print(f"Error: Prompt templates file not found at {PROMPTS_FILE}")
        sys.exit(1)
    
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
            return data.get("templates", {})
        except Exception as e:
            print(f"Error parsing templates YAML: {e}")
            sys.exit(1)


def find_runs() -> List[Path]:
    """Finds all run_* directories in reports/ sorted by creation time (latest first)."""
    if not REPORTS_DIR.exists():
        print(f"Error: Reports directory {REPORTS_DIR} does not exist.")
        sys.exit(1)
        
    runs = [d for d in REPORTS_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")]
    # Sort by directory name (which starts with timestamp YYYYMMDD_HHMMSS) in reverse order (latest first)
    runs.sort(key=lambda x: x.name, reverse=True)
    return runs


def get_metrics_summary(run_dir: Path) -> str:
    """Reads or generates metrics_summary.md for the given run directory."""
    summary_md_path = run_dir / "metrics_summary.md"
    
    # Check if summary already exists
    if summary_md_path.exists():
        with open(summary_md_path, "r", encoding="utf-8") as f:
            return f.read()
            
    # Auto-generate if missing but result_metrics.yaml exists
    result_metrics_path = run_dir / "result_metrics.yaml"
    if result_metrics_path.exists():
        parse_script = SCRIPT_DIR / "parse_metrics.py"
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
        
    fallback_path = REPORTS_DIR / "metrics_summary.md"
    if fallback_path.exists():
        with open(fallback_path, "r", encoding="utf-8") as f:
            return f.read()
            
    print(f"Error: Could not find any metrics summary for run {run_dir.name}")
    sys.exit(1)


def copy_to_clipboard(text: str) -> bool:
    """Copies the text to clipboard. Supports macOS pbcopy, and falls back to other tools if needed."""
    try:
        # On macOS, pbcopy is native and always available
        if sys.platform == "darwin":
            process = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            process.communicate(input=text.encode("utf-8"))
            return True
        else:
            # Fallback for Linux (xclip/xsel) or Windows (clip)
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


def display_runs_table(runs: List[Path]):
    """Renders a beautiful table of available runs."""
    if HAS_RICH:
        table = Table(title="Available Benchmarking Runs", show_header=True, header_style="bold magenta")
        table.add_column("Index", style="cyan", justify="right")
        table.add_column("Run Directory Name", style="white")
        table.add_column("Created At (Estimate)", style="green")
        
        for idx, run in enumerate(runs):
            # Try parsing timestamp from folder name (run_YYYYMMDD_HHMMSS_model)
            parts = run.name.split("_")
            time_str = "Unknown"
            if len(parts) >= 3:
                date_part = parts[1]
                time_part = parts[2]
                if len(date_part) == 8 and len(time_part) == 6:
                    time_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]} {time_part[:2]}:{time_part[2:4]}:{time_part[4:]}"
            table.add_row(str(idx + 1), run.name, time_str)
        console.print(table)
    else:
        print("\nAvailable Benchmarking Runs:")
        print(f"{'Index':<6} | {'Run Directory Name':<50}")
        print("-" * 60)
        for idx, run in enumerate(runs):
            print(f"{idx + 1:<6} | {run.name:<50}")


def display_templates_table(templates: Dict[str, Any]):
    """Renders a table of available templates."""
    if HAS_RICH:
        table = Table(title="Prompt Templates", show_header=True, header_style="bold blue")
        table.add_column("Key", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Description", style="green")
        
        for key, template in templates.items():
            table.add_row(key, template.get("name", key), template.get("description", ""))
        console.print(table)
    else:
        print("\nAvailable Prompt Templates:")
        print(f"{'Key':<20} | {'Name':<40} | {'Description'}")
        print("-" * 80)
        for key, template in templates.items():
            print(f"{key:<20} | {template.get('name', key):<40} | {template.get('description', '')}")


def interactive_selection(runs: List[Path], templates: Dict[str, Any]) -> tuple:
    """Prompts the user interactively to select run(s) and template."""
    display_runs_table(runs)
    
    # 1. Select run(s)
    run1_idx = 1
    if len(runs) > 1:
        if HAS_RICH:
            run1_idx = IntPrompt.ask("Select run index (default: latest)", choices=[str(i) for i in range(1, len(runs) + 1)], default=1)
        else:
            try:
                val = input("Select run index (default: 1 [latest]): ").strip()
                run1_idx = int(val) if val else 1
            except ValueError:
                run1_idx = 1
    
    run1 = runs[run1_idx - 1]
    
    # 2. Select Template
    display_templates_table(templates)
    
    template_keys = list(templates.keys())
    selected_template_key = template_keys[0]
    if HAS_RICH:
        selected_template_key = Prompt.ask("Select prompt template key", choices=template_keys, default=template_keys[0])
    else:
        val = input(f"Select prompt template key (default: {template_keys[0]}): ").strip()
        if val in templates:
            selected_template_key = val

    run2 = None
    # 3. If comparison template is chosen, select a second run
    if "compare" in selected_template_key:
        if len(runs) < 2:
            print("Error: You selected a comparison template, but there is only one run available.")
            sys.exit(1)
            
        if HAS_RICH:
            run2_idx = IntPrompt.ask("Select second run index for comparison", choices=[str(i) for i in range(1, len(runs) + 1) if i != run1_idx], default=2 if run1_idx != 2 else 1)
        else:
            try:
                val = input("Select second run index for comparison: ").strip()
                run2_idx = int(val) if val else (2 if run1_idx != 2 else 1)
            except ValueError:
                run2_idx = 2 if run1_idx != 2 else 1
        run2 = runs[run2_idx - 1]

    return run1, run2, selected_template_key


def main():
    parser = argparse.ArgumentParser(description="Copy RAG results prompt to clipboard")
    parser.add_argument("--run", "-r", type=str, help="First run directory name, absolute path, or 1-based index (latest is 1).")
    parser.add_argument("--run2", "-r2", type=str, help="Second run directory name, absolute path, or index (for comparison).")
    parser.add_argument("--template", "-t", type=str, help="Template key to use.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Always run in interactive mode.")
    parser.add_argument("--print-only", "-p", action="store_true", help="Only print the prompt to stdout, do not copy to clipboard.")
    args = parser.parse_args()

    runs = find_runs()
    if not runs:
        print("Error: No runs found in reports/ directory.")
        sys.exit(1)

    templates = load_templates()
    
    # Determine mode
    is_interactive = args.interactive or (not args.run and not args.template)
    
    if is_interactive:
        if HAS_RICH:
            console.print(Panel("[bold green]Welcome to RAG Prompt Exporter![/bold green]\nSelect runs and templates to generate a formatted prompt."))
        else:
            print("=== RAG Prompt Exporter ===")
        run1, run2, template_key = interactive_selection(runs, templates)
    else:
        # Resolve run 1
        run1 = None
        if args.run:
            if args.run.isdigit():
                idx = int(args.run)
                if 1 <= idx <= len(runs):
                    run1 = runs[idx - 1]
            else:
                # Check direct path or name
                path_opt = Path(args.run)
                if path_opt.is_absolute() and path_opt.exists():
                    run1 = path_opt
                elif (REPORTS_DIR / args.run).exists():
                    run1 = REPORTS_DIR / args.run
        if not run1:
            run1 = runs[0]  # Default to latest
            
        # Resolve template
        template_key = args.template
        if not template_key or template_key not in templates:
            template_key = list(templates.keys())[0]
            
        # Resolve run 2 (if comparison)
        run2 = None
        if "compare" in template_key:
            if args.run2:
                if args.run2.isdigit():
                    idx = int(args.run2)
                    if 1 <= idx <= len(runs):
                        run2 = runs[idx - 1]
                else:
                    path_opt = Path(args.run2)
                    if path_opt.is_absolute() and path_opt.exists():
                        run2 = path_opt
                    elif (REPORTS_DIR / args.run2).exists():
                        run2 = REPORTS_DIR / args.run2
            if not run2 and len(runs) > 1:
                # Default to second latest
                run2 = runs[1]

    # Load template configuration
    tpl = templates[template_key]
    system_prompt = tpl.get("system_prompt", "")
    user_prompt_template = tpl.get("user_prompt_template", "")

    # Load summaries
    summary1 = get_metrics_summary(run1)
    
    # Formulate user prompt
    if "compare" in template_key:
        if not run2:
            print("Error: Comparison template selected but second run is missing.")
            sys.exit(1)
        summary2 = get_metrics_summary(run2)
        user_prompt = user_prompt_template.format(
            metrics_summary_1=summary1,
            metrics_summary_2=summary2
        )
        run_info_str = f"Comparing: {run1.name} VS {run2.name}"
    else:
        user_prompt = user_prompt_template.format(
            metrics_summary=summary1
        )
        run_info_str = f"Selected Run: {run1.name}"

    # Construct the full text (with system instructions if desired, or just the user prompt)
    full_prompt_text = f"System Instruction:\n{system_prompt}\n\nUser Message:\n{user_prompt}"

    # Copy to clipboard
    copied = False
    if not args.print_only:
        copied = copy_to_clipboard(full_prompt_text)

    # Show results
    if HAS_RICH:
        console.print(Panel(f"[bold green]Prompt Generated Successfully![/bold green]\n[cyan]{run_info_str}[/cyan]\n[bold]Template:[/bold] {tpl.get('name')}", title="Success"))
        
        # Display preview of what was copied
        console.print(Panel(full_prompt_text[:1000] + ("\n... [TRUNCATED PREVIEW] ..." if len(full_prompt_text) > 1000 else ""), title="Prompt Preview", border_style="dim"))
        
        if copied:
            console.print("[bold green][+] The complete prompt has been copied to your clipboard![/bold green] You can now paste it into Claude or ChatGPT.")
        else:
            console.print("[yellow][!] Failed to copy to clipboard automatically. Please copy the text from the terminal manually.[/yellow]")
            console.print("\n=== FULL PROMPT START ===")
            print(full_prompt_text)
            console.print("=== FULL PROMPT END ===\n")
    else:
        print("\n" + "="*50)
        print(f"Success! {run_info_str}")
        print(f"Template: {tpl.get('name')}")
        print("="*50)
        print("Prompt Preview (first 500 chars):")
        print(full_prompt_text[:500] + "\n...")
        print("="*50)
        if copied:
            print("[+] Prompt copied to clipboard successfully!")
        else:
            print("[-] Could not copy to clipboard. Full prompt is displayed below:")
            print("\n=== FULL PROMPT START ===")
            print(full_prompt_text)
            print("=== FULL PROMPT END ===\n")


if __name__ == "__main__":
    main()
