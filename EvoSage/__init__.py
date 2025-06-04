import logging

logger = logging.getLogger(__name__)

_DEF_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def setup_logging(level: str | int = logging.INFO) -> None:
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=level, format=_DEF_FORMAT)
    logger.setLevel(level)

