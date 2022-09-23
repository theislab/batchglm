#!/usr/bin/env python
"""Command-line interface."""
import click
from rich import traceback


@click.command()
@click.version_option(version="0.7.4", message=click.style("batchglm Version: 0.7.4"))
def main() -> None:
    """batchglm."""


if __name__ == "__main__":
    traceback.install()
    main(prog_name="batchglm")  # pragma: no cover
