from typing import Callable

import click


def acbm_cli(c: Callable):
    @click.command()
    @click.option(
        "--config_file", prompt="Filepath relative to repo root of config", type=str
    )
    def main(config_file):
        return c(config_file)

    return main
