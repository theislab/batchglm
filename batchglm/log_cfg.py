import sys

import logging

logger = logging.getLogger('.'.join(__name__.split('.')[:-1]))

_is_interactive = bool(getattr(sys, 'ps1', sys.flags.interactive))


def unconfigure_logging():
    if logger.hasHandlers():
        for handler in logger.handlers:
            logger.removeHandler(handler)

    logger.setLevel(logging.NOTSET)


def setup_logging(verbosity="WARNING", stream=None, format=logging.BASIC_FORMAT):
    unconfigure_logging()

    if isinstance(verbosity, str):
        verbosity = getattr(logging, verbosity)

    logger.setLevel(verbosity)

    if stream is not None:
        if isinstance(stream, str):
            if stream.lower() == "stdout":
                stream = sys.stdout
            elif stream.lower() == "stderr":
                stream = sys.stderr
            else:
                raise ValueError("Unknown stream %s" % stream)

        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter(format, None))
        logger.addHandler(handler)

# If we are in an interactive environment (like Jupyter), set loglevel to INFO and pipe the output to stdout.
if _is_interactive:
    setup_logging(logging.INFO)
else:
    setup_logging(logging.WARNING)
