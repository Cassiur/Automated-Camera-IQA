"""Top-level Click command group for the IQA CLI."""

import click

from iqa.cli.cmd_boresight import boresight
from iqa.cli.cmd_isp import isp
from iqa.cli.cmd_sfr import sfr


@click.group()
@click.version_option(package_name="automated-camera-iqa")
def cli() -> None:
    """
    Automated-Camera-IQA — automotive camera ISP simulation & quality assessment.

    \b
    Sub-commands:
      isp        Run ISP pipeline (BLC → Demosaic → AWB → CCM)
      sfr        Evaluate SFR/MTF50 via ISO 12233 slanted-edge method
      boresight  Check multi-camera boresight alignment consistency
    """


cli.add_command(isp)
cli.add_command(sfr)
cli.add_command(boresight)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
