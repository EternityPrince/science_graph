#!/usr/bin/env python3
"""
Science Graph — Copy Benchmark Run Results Prompt to Clipboard.
Finds benchmarking runs in reports/, formats the metrics summary,
injects it into a prompt template, and copies the result to the clipboard.
"""

import sys
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Any

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, IntPrompt
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False

# Set up python path to resolve src and core imports correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.clipboard import (
    copy_to_clipboard,
    find_runs,
    get_metrics_summary,
    load_templates as _core_load_templates,
    display_runs_table,
    display_templates_table,
    interactive_selection,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = SCRIPT_DIR / "reports"
PROMPTS_FILE = SCRIPT_DIR / "prompts" / "analysis_prompts.yaml"


def load_templates() -> Dict[str, Any]:
    """Loads prompt templates from the default yaml file."""
    return _core_load_templates(PROMPTS_FILE)



def main():
    parser = argparse.ArgumentParser(description="Copy RAG results prompt to clipboard")
    parser.add_argument("--run", "-r", type=str, help="First run directory name, absolute path, or 1-based index (latest is 1).")
    parser.add_argument("--run2", "-r2", type=str, help="Second run directory name, absolute path, or index (for comparison).")
    parser.add_argument("--template", "-t", type=str, help="Template key to use.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Always run in interactive mode.")
    parser.add_argument("--print-only", "-p", action="store_true", help="Only print the prompt to stdout, do not copy to clipboard.")
    args = parser.parse_args()

    runs = find_runs(REPORTS_DIR)
    if not runs:
        print("Error: No runs found in reports/ directory.")
        sys.exit(1)

    templates = load_templates()
    is_interactive = args.interactive or (not args.run and not args.template)
    
    if is_interactive:
        if HAS_RICH:
            console.print(Panel("[bold green]Welcome to RAG Prompt Exporter![/bold green]\nSelect runs and templates to generate a formatted prompt."))
        else:
            print("=== RAG Prompt Exporter ===")
        run1, run2, template_key = interactive_selection(runs, templates)
    else:
        run1 = None
        if args.run:
            if args.run.isdigit():
                idx = int(args.run)
                if 1 <= idx <= len(runs):
                    run1 = runs[idx - 1]
            else:
                path_opt = Path(args.run)
                if path_opt.is_absolute() and path_opt.exists():
                    run1 = path_opt
                elif (REPORTS_DIR / args.run).exists():
                    run1 = REPORTS_DIR / args.run
        if not run1:
            run1 = runs[0]
            
        template_key = args.template
        if not template_key or template_key not in templates:
            template_key = list(templates.keys())[0]
            
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
                run2 = runs[1]

    tpl = templates[template_key]
    system_prompt = tpl.get("system_prompt", "")
    user_prompt_template = tpl.get("user_prompt_template", "")

    summary1 = get_metrics_summary(run1, REPORTS_DIR, SCRIPT_DIR)
    
    if "compare" in template_key:
        if not run2:
            print("Error: Comparison template selected but second run is missing.")
            sys.exit(1)
        summary2 = get_metrics_summary(run2, REPORTS_DIR, SCRIPT_DIR)
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

    full_prompt_text = f"System Instruction:\n{system_prompt}\n\nUser Message:\n{user_prompt}"

    copied = False
    if not args.print_only:
        copied = copy_to_clipboard(full_prompt_text)

    if HAS_RICH:
        console.print(Panel(f"[bold green]Prompt Generated Successfully![/bold green]\n[cyan]{run_info_str}[/cyan]\n[bold]Template:[/bold] {tpl.get('name')}", title="Success"))
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
