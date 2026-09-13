import logging


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# We remove info logs for the below loggers in combination, because during the biological
# evaluation, they produce a lot of info logs that are not useful for the user and clutter the output.
# For example "... storing 'cell_type' as categorical" or "... storing 'cluster' as categorical".
NOISY_LOGGERS = ("anndata", "scanpy")


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        force=True,
    )

    exclude_noisy_loggers()


def exclude_noisy_loggers() -> None:
    for logger_name in NOISY_LOGGERS:
        # messages below this severity will get dropped
        logging.getLogger(logger_name).setLevel(logging.WARNING)
