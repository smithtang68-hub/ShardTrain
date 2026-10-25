import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


class RankJsonFormatter(logging.Formatter):
    def __init__(self, rank: int) -> None:
        super().__init__()
        self.rank = rank

    def format(self, record: logging.LogRecord) -> str:
        data = {
            "time_utc": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "rank": self.rank,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


def make_rank_logger(directory: Path, rank: int, level: str = "INFO") -> logging.Logger:
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"shardtrain.rank{rank}")
    close_logger(logger)
    logger.setLevel(level)
    logger.propagate = False
    file_handler = logging.FileHandler(directory / f"rank{rank}.jsonl", encoding="utf-8")
    file_handler.setFormatter(RankJsonFormatter(rank))
    logger.addHandler(file_handler)
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level if rank == 0 else logging.ERROR)
    console.setFormatter(logging.Formatter(f"[rank {rank}] %(levelname)s %(message)s"))
    logger.addHandler(console)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
