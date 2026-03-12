#!/usr/bin/env python3
"""CLI entry point for the OCR past-paper pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

load_dotenv()  # Load .env file if present

console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="ocr-pipeline")
def cli() -> None:
    """OCR Pipeline – Convert past papers into structured JSON."""
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "-o", "--output",
    type=click.Path(),
    default=None,
    help="Output JSON file path. Defaults to <input_stem>_output/<input_stem>.json",
)
@click.option(
    "--provider",
    type=click.Choice(["openai", "anthropic"], case_sensitive=False),
    default="openai",
    help="LLM provider to use (default: openai)",
)
@click.option(
    "--model",
    default=None,
    help="Model name (default: gpt-4o for openai, claude-sonnet-4 for anthropic)",
)
@click.option(
    "--api-key",
    default=None,
    help="API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var)",
)
@click.option(
    "--save-pages",
    is_flag=True,
    default=False,
    help="Save individual page images alongside the JSON output",
)
@click.option(
    "--instructions",
    default="",
    help="Extra instructions for the LLM (e.g. 'This is a chemistry paper')",
)
@click.option(
    "--pretty/--compact",
    default=True,
    help="Pretty-print the JSON output (default: pretty)",
)
def scan(
    input_file: str,
    output: str | None,
    provider: str,
    model: str | None,
    api_key: str | None,
    save_pages: bool,
    instructions: str,
    pretty: bool,
) -> None:
    """Scan a past paper (PDF or image) and extract questions as JSON."""
    from pipeline import run_pipeline

    input_path = Path(input_file)

    console.print(
        Panel(
            f"[bold cyan]Scanning:[/bold cyan] {input_path.name}\n"
            f"[dim]Provider: {provider} | Format: {input_path.suffix.upper()}[/dim]",
            title="OCR Pipeline",
            border_style="blue",
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading and processing file...", total=None)

        try:
            result = run_pipeline(
                input_path,
                output_path=output,
                provider=provider,
                api_key=api_key,
                model=model,
                extra_instructions=instructions,
                save_page_images=save_pages,
            )
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Unexpected error:[/red] {e}")
            sys.exit(1)

        progress.update(task, description="Done!")

    # Display summary
    _print_summary(result)

    # Show output location
    if output:
        out_path = output
    else:
        out_path = str(input_path.parent / f"{input_path.stem}_output" / f"{input_path.stem}.json")
    console.print(f"\n[green]JSON saved to:[/green] {out_path}")


@cli.command()
@click.argument("input_files", nargs=-1, type=click.Path(exists=True), required=True)
@click.option(
    "-o", "--output-dir",
    type=click.Path(),
    required=True,
    help="Directory to save all JSON outputs",
)
@click.option(
    "--provider",
    type=click.Choice(["openai", "anthropic"], case_sensitive=False),
    default="openai",
    help="LLM provider to use",
)
@click.option("--model", default=None, help="Model name")
@click.option("--api-key", default=None, help="API key")
@click.option("--save-pages", is_flag=True, default=False, help="Save page images")
@click.option("--instructions", default="", help="Extra instructions for the LLM")
def batch(
    input_files: tuple[str, ...],
    output_dir: str,
    provider: str,
    model: str | None,
    api_key: str | None,
    save_pages: bool,
    instructions: str,
) -> None:
    """Batch process multiple past papers."""
    from pipeline import run_pipeline

    console.print(
        Panel(
            f"[bold cyan]Batch processing {len(input_files)} file(s)[/bold cyan]",
            title="OCR Pipeline – Batch Mode",
            border_style="blue",
        )
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, input_file in enumerate(input_files, 1):
        input_path = Path(input_file)
        json_out = output_path / f"{input_path.stem}.json"
        console.print(f"\n[cyan]({i}/{len(input_files)})[/cyan] Processing: {input_path.name}")

        try:
            result = run_pipeline(
                input_path,
                output_path=json_out,
                provider=provider,
                api_key=api_key,
                model=model,
                extra_instructions=instructions,
                save_page_images=save_pages,
            )
            _print_summary(result)
            console.print(f"  [green]Saved:[/green] {json_out}")
        except Exception as e:
            console.print(f"  [red]Failed:[/red] {e}")


@cli.command("generate-notes")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "-o", "--output",
    type=click.Path(),
    default=None,
    help="Output JSON file path. Defaults to <input_stem>_notebook/<input_stem>_notebook.json",
)
@click.option(
    "--papers",
    multiple=True,
    type=click.Path(exists=True),
    help="Past paper JSON files to match questions from (can be repeated)",
)
@click.option(
    "--provider",
    type=click.Choice(["openai", "anthropic"], case_sensitive=False),
    default="openai",
    help="LLM provider to use (default: openai)",
)
@click.option("--model", default=None, help="Model name")
@click.option("--api-key", default=None, help="API key")
@click.option("--save-pages", is_flag=True, default=False, help="Save page images")
@click.option(
    "--instructions",
    default="",
    help="Extra instructions for the LLM (e.g. 'This is a Year 2 Microeconomics lecture')",
)
def generate_notes(
    input_file: str,
    output: str | None,
    papers: tuple[str, ...],
    provider: str,
    model: str | None,
    api_key: str | None,
    save_pages: bool,
    instructions: str,
) -> None:
    """Generate an intuitive study guide from lecture slides/notes (PDF or image).

    Optionally match past paper questions to the generated notes by providing
    --papers flags with paths to previously extracted paper JSONs.

    Examples:

        python main.py generate-notes lecture_slides.pdf

        python main.py generate-notes slides.pdf --papers paper1.json --papers paper2.json

        python main.py generate-notes notes.pdf -o output/guide.json --instructions "Year 2 Macro"
    """
    from pipeline import run_notebook_pipeline

    input_path = Path(input_file)

    console.print(
        Panel(
            f"[bold cyan]Generating study guide from:[/bold cyan] {input_path.name}\n"
            f"[dim]Provider: {provider} | Format: {input_path.suffix.upper()}[/dim]"
            + (f"\n[dim]Matching against {len(papers)} paper(s)[/dim]" if papers else ""),
            title="📓 Notebook Generator",
            border_style="green",
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing lecture content...", total=None)

        try:
            result = run_notebook_pipeline(
                input_path,
                output_path=output,
                paper_paths=list(papers) if papers else None,
                provider=provider,
                api_key=api_key,
                model=model,
                extra_instructions=instructions,
                save_page_images=save_pages,
            )
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Unexpected error:[/red] {e}")
            sys.exit(1)

        progress.update(task, description="Done!")

    # Display notebook summary
    _print_notebook_summary(result)

    # Show output location
    if output:
        out_path = output
    else:
        out_path = str(input_path.parent / f"{input_path.stem}_notebook" / f"{input_path.stem}_notebook.json")
    console.print(f"\n[green]Notebook JSON saved to:[/green] {out_path}")


@cli.command()
@click.argument("json_file", type=click.Path(exists=True))
def show(json_file: str) -> None:
    """Display a previously extracted JSON file in a readable format."""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    _print_summary(data)

    # Also show formatted JSON
    formatted = json.dumps(data, indent=2, ensure_ascii=False)
    syntax = Syntax(formatted, "json", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="Full JSON", border_style="dim"))


def _print_notebook_summary(data: dict) -> None:
    """Print a summary of the generated notebook."""
    sections = data.get("sections", [])

    table = Table(
        title=f"📓 {data.get('title', 'Study Guide')}",
        show_lines=True,
    )
    table.add_column("#", style="bold", width=4)
    table.add_column("Icon", width=4)
    table.add_column("Section", style="cyan", max_width=40)
    table.add_column("Tags", style="yellow", max_width=40)
    table.add_column("Subsections", style="dim", width=12)

    for i, section in enumerate(sections, 1):
        tags = ", ".join(section.get("tags", [])[:5])
        if len(section.get("tags", [])) > 5:
            tags += "..."
        sub_count = len(section.get("subsections", []) or [])
        table.add_row(
            str(i),
            section.get("icon", ""),
            section.get("title", ""),
            tags,
            str(sub_count),
        )

    console.print(table)
    console.print(
        f"  [dim]Course: {data.get('course', 'N/A')} | "
        f"Sections: {len(sections)} | "
        f"Matched questions: {data.get('questionCount', 0)}[/dim]"
    )


def _print_summary(data: dict) -> None:
    """Print a nice summary table of the extracted questions."""
    questions = data.get("questions", [])

    table = Table(title=f"📄 {data.get('title', 'Exam Paper')}", show_lines=True)
    table.add_column("#", style="bold", width=4)
    table.add_column("Type", style="cyan", width=18)
    table.add_column("Question (preview)", style="white", max_width=60)
    table.add_column("Equation", style="yellow", width=12)
    table.add_column("Images", style="magenta", width=10)

    mc_count = 0
    oe_count = 0

    for q in questions:
        qtype = q.get("type", "unknown")
        if qtype == "multiple-choice":
            mc_count += 1
        else:
            oe_count += 1

        text_preview = q.get("text", "")[:80]
        if len(q.get("text", "")) > 80:
            text_preview += "..."

        has_eq = "Yes" if q.get("equation") else "-"
        has_img = "Yes" if q.get("images") else "-"

        table.add_row(
            str(q.get("number", "?")),
            qtype,
            text_preview,
            has_eq,
            has_img,
        )

    console.print(table)
    console.print(
        f"  [dim]Total: {len(questions)} questions "
        f"({mc_count} multiple-choice, {oe_count} open-ended)[/dim]"
    )


if __name__ == "__main__":
    cli()
