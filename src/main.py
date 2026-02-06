import click
import logging
from typing import Tuple
from src.config import config, AppConfig  # Import both
from src.utils.app_logger import AppLogger

# --- Import pipeline runners ---
try:
    from src.pipelines import data_ingestion

    # Add others as you implement them:
    # from src.pipelines import data_cleaning, feature_engineering
except ImportError as e:
    logging.critical(
        f"Failed to import a pipeline module: {e}. Ensure all pipeline scripts exist.",
        exc_info=True,
    )
    exit(1)


# --- Main CLI Group ---
@click.group()
@click.pass_context
def cli(ctx):
    """Liquid Veins Algo Trading Pipeline Orchestrator"""
    ctx.ensure_object(dict)

    # Use the global config instance we created in config.py
    ctx.obj["config"] = config

    # Initialize logger using the verified log directory
    ctx.obj["logger"] = AppLogger(
        "MainCLI", log_dir=config.dirs["logs"], clear_existing=False
    )

    # Set logging level based on debug_mode in YAML
    log_level = logging.DEBUG if config.global_settings.debug_mode else logging.INFO
    logging.basicConfig(level=log_level)


# --- Pipeline Definitions ---
PIPELINES = {
    "ingest": data_ingestion,
    # "clean": data_cleaning,
}


def _print_summary(status_dict: dict):
    """Prints a formatted summary table of pipeline statuses."""
    click.echo("\n" + "=" * 50)
    click.echo("PIPELINE EXECUTION SUMMARY".center(50))
    click.echo("=" * 50)
    for name, status in status_dict.items():
        color = (
            "green"
            if status == "Success"
            else "red" if status == "Failed" else "yellow"
        )
        click.echo(f"- {name:<20} {click.style(status, fg=color)}")
    click.echo("=" * 50)


# --- CLI Commands ---
@cli.command()
@click.argument("pipelines_to_run", nargs=-1, type=click.Choice(list(PIPELINES.keys())))
@click.pass_context
def run(ctx, pipelines_to_run: Tuple[str]):
    """Run specified pipelines: python -m src.main run ingest"""
    logger = ctx.obj["logger"]

    if not pipelines_to_run:
        click.echo("Error: No pipelines specified.")
        return

    pipeline_statuses = {name: "Skipped" for name in PIPELINES}

    for name in pipelines_to_run:
        module = PIPELINES.get(name)
        if module and hasattr(module, "run"):
            logger.info(f">>> Starting Pipeline: {name}")
            try:
                module.run()
                pipeline_statuses[name] = "Success"
            except Exception as e:
                pipeline_statuses[name] = "Failed"
                logger.error(f"Pipeline '{name}' failed: {e}")
                _print_summary(pipeline_statuses)
                exit(1)
        else:
            pipeline_statuses[name] = "Not Implemented"

    _print_summary(pipeline_statuses)


@cli.command()
@click.pass_context
def run_all(ctx):
    """Run all pipelines in sequence."""
    ctx.invoke(run, pipelines_to_run=tuple(PIPELINES.keys()))


@cli.command()
def show_config():
    """Verify and pretty-print the current configuration."""
    config.pretty_print()


if __name__ == "__main__":
    cli(obj={})
