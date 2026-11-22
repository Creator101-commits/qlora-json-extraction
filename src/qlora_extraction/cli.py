"""Command-line interface for the QLoRA JSON extraction benchmark."""

from __future__ import annotations

import typer

from qlora_extraction import __version__

app = typer.Typer(
    help="QLoRA JSON Extraction CLI.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print package version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
