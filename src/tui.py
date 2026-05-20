import sys
from typing import List, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.align import Align

from src.rag import RAGPipeline

class TUIHistory:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.turns: List[Tuple[str, str]] = []

    def add_turn(self, query: str, response: str):
        self.turns.append((query, response))
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def format_for_llm(self) -> str:
        formatted = []
        for q, r in self.turns:
            formatted.append(f"<|im_start|>user\n{q}\n<|im_end|>\n<|im_start|>assistant\n{r}\n<|im_end|>\n")
        return "".join(formatted)


def run_tui_chat(rag_pipeline: RAGPipeline):
    console = Console()
    
    # 1. Print Welcome Banner
    console.clear()
    
    welcome_text = Text()
    welcome_text.append("🧠 PDF Graph Analyzer — Interactive RAG Chat\n", style="bold magenta")
    welcome_text.append("Ask questions about your scientific papers and their graph relationships.\n", style="italic")
    welcome_text.append("Commands: 'exit' or 'quit' to end session, 'clear' to clear screen.\n", style="dim")
    
    # Try fetching stats
    try:
        stats = rag_pipeline.graph_repo.get_stats()
        stats_str = f"Database Stats: {stats.get('papers', 0)} papers, {stats.get('authors', 0)} authors, {stats.get('concepts', 0)} concepts, {stats.get('edges', 0)} connections"
        welcome_text.append(f"\n{stats_str}", style="green")
    except Exception:
        pass
        
    console.print(Panel(Align.center(welcome_text), border_style="bold blue", padding=(1, 2)))
    
    history = TUIHistory(max_turns=5)
    
    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]Query[/bold cyan]")
            
            # Sanitise
            query = user_input.strip()
            if not query:
                continue
                
            if query.lower() in ("exit", "quit"):
                console.print("[bold red]Ending chat session. Goodbye![/bold red]")
                break
                
            if query.lower() == "clear":
                console.clear()
                console.print(Panel(Align.center(welcome_text), border_style="bold blue", padding=(1, 2)))
                continue
            
            # Show processing spinner
            with console.status("[bold yellow]Retrieving context & generating answer...[/bold yellow]", spinner="dots"):
                history_str = history.format_for_llm()
                response = rag_pipeline.ask(query, limit=5, history_str=history_str)
                
            # Add to history
            history.add_turn(query, response)
            
            # Print response
            console.print("\n")
            console.print(
                Panel(
                    Markdown(response),
                    title="[bold green]Assistant[/bold green]",
                    title_align="left",
                    border_style="green",
                    padding=(1, 2)
                )
            )
            
        except KeyboardInterrupt:
            console.print("\n[bold red]Session interrupted. Goodbye![/bold red]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
            
if __name__ == "__main__":
    # For quick testing, run tui directly
    from src.cli import get_services
    graph, vector, emb, llm = get_services(load_llm=True)
    rag = RAGPipeline(graph, vector, emb, llm)
    run_tui_chat(rag)
