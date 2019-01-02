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

# TODO add extra info logger to unit tests that only output unit tests info logs
#INFO_UNIT_TESTS_NUM = 25
#logging.addLevelName(INFO_UNIT_TESTS_NUM, "INFO_UNIT_TESTS")
#def info_unit_tests(self, message, *args, **kws):
#    if self.isEnabledFor(INFO_UNIT_TESTS_NUM):
#        self._log(INFO_UNIT_TESTS_NUM, message, args, **kws)
#logging.Logger.info_unit_tests = info_unit_tests


# If we are in an interactive environment (like Jupyter), set loglevel to INFO and pipe the output to stdout.
if _is_interactive:
    setup_logging(logging.INFO)
else:
    setup_logging(logging.WARNING)
