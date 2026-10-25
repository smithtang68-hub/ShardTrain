import logging
from pathlib import Path

from sharded_runtime.config import Config, ConfigError
from sharded_runtime.distributed.context import DistributedContext, WorkerEnvironment
from sharded_runtime.distributed.nccl_ordering import prepare_nccl_launch_order
from sharded_runtime.observability.logging import close_logger, make_rank_logger
from sharded_runtime.observability.metadata import save_metadata


class WorkerSession:
    """Own the logger and distributed context; release both on every exit path."""

    def __init__(self, config: Config, mode: str) -> None:
        self.config = config
        self.mode = mode
        if not config.debug.output_dir.strip():
            raise ConfigError("debug.output_dir must not be empty")
        self.directory = Path(config.debug.output_dir)
        self.context: DistributedContext | None = None
        self.logger: logging.Logger | None = None

    def __enter__(self) -> "WorkerSession":
        log_level = self.config.debug.log_level
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            log_level = "INFO"  # Keep a diagnostic log for the validator's level error.
        # Create a diagnostic log even for malformed launcher rank values.
        try:
            self.worker = WorkerEnvironment.read()
        except ConfigError:
            logger = make_rank_logger(self.directory, 0, log_level)
            logger.exception("Worker environment validation failed")
            close_logger(logger)
            raise
        self.logger = make_rank_logger(self.directory, self.worker.rank, log_level)
        try:
            self.config.validate(self.worker.world_size, mode=self.mode)
            if (
                self.worker.launched
                and self.worker.world_size > 1
                and self.config.parallel.backend == "nccl"
                and self.config.runtime.communication_policy == "stream_ordered"
            ):
                contract = prepare_nccl_launch_order()
                self.logger.info("NCCL host launch ordering contract: %s", contract)
            self.context = DistributedContext(
                self.config.parallel,
                self.worker,
                self.config.debug.distributed_timeout_seconds,
                self.logger,
            )
            save_metadata(self.directory, self.config, self.worker, self.context.device)
            self.logger.info("Resolved config: %s", self.config.to_dict())
            return self
        except BaseException:
            self.logger.exception("Worker bootstrap failed")
            if self.context is not None:
                self.context.close()
            close_logger(self.logger)
            raise

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if exc_type is not None:
                self.logger.error("Worker failed", exc_info=(exc_type, exc, traceback))
            if self.context is not None:
                self.context.close()
                if exc_type is None and self.context.cleanup_errors:
                    raise RuntimeError(
                        "Worker cleanup failed: " + "; ".join(self.context.cleanup_errors)
                    )
        finally:
            if self.logger is not None:
                close_logger(self.logger)
