# The MIT License (MIT)
# Subject to the terms and conditions contained in LICENSE
import logging
import os

logger = logging.getLogger(__name__)


def initialize_logger(to_file=False, log_name=None, output_dir=None):
    """
    Adapted from

    https://aykutakin.wordpress.com/2013/08/06/logging-to-console-and-file-in-python/

    This logs to the console and to files in the content root directory.

    Args:
        to_file (bool): if True, write a lot file

        log_name (str): prefixes the name of the log file

        output_dir (str): (optional) where to write the log file; if `None`,
            logging will be to the content root directory

    Returns:
        None

    """

    log_fmt = (
        "[%(asctime)s%(levelname)8s], [%(filename)s:%(lineno)s "
        + "- %(funcName)s()], %(message)s"
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(log_fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create error file handler and set level to error
    if to_file:
        log_name = log_name.replace(" ", "-").lower()

        if output_dir is None:
            output_dir = os.path.dirname(os.path.realpath(__file__))

        elif not os.path.isdir(output_dir):
            raise FileNotFoundError(f"no directory {output_dir}")
        # create a file handler and set level to INFO
        logger.info(f"log file : {log_name}")

        handler = logging.FileHandler(os.path.join(output_dir, log_name), "w")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(log_fmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
