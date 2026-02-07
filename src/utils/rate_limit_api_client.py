import time
from datetime import datetime
from collections import deque
from typing import Optional, Callable, Any
from src.utils.app_logger import AppLogger


class RateLimitedApiClient:
    """Wraps any API client with aggressive rate limiting diagnostics."""

    def __init__(
        self,
        client,
        max_requests_per_minute: int = 200,
        logger: Optional[AppLogger] = None,
    ):
        self.client = client
        self.max_requests = max_requests_per_minute
        self.logger = logger or AppLogger(name="RateLimitedApiClient")
        self.timestamps = deque()
        self.window_seconds = 60

        self.logger.info(
            "Initialized RateLimitedApiClient",
            extra={
                "max_requests_per_minute": self.max_requests,
                "window_seconds": self.window_seconds,
                "client_type": type(client).__name__,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune_old_requests(self):
        now = datetime.utcnow()
        before = len(self.timestamps)

        self.logger.debug(
            "Pruning old requests (before)",
            extra={
                "queue_size": before,
                "timestamps": [ts.isoformat() for ts in self.timestamps],
                "now": now.isoformat(),
            },
        )

        while (
            self.timestamps
            and (now - self.timestamps[0]).total_seconds() > self.window_seconds
        ):
            removed = self.timestamps.popleft()
            self.logger.debug(
                "Removed expired request timestamp",
                extra={
                    "removed_timestamp": removed.isoformat(),
                    "age_seconds": (now - removed).total_seconds(),
                },
            )

        after = len(self.timestamps)

        self.logger.debug(
            "Pruning complete",
            extra={
                "queue_size_before": before,
                "queue_size_after": after,
            },
        )

    def _should_throttle(self) -> bool:
        self._prune_old_requests()

        should = len(self.timestamps) >= self.max_requests

        self.logger.debug(
            "Throttle check",
            extra={
                "current_requests": len(self.timestamps),
                "max_requests": self.max_requests,
                "should_throttle": should,
            },
        )

        return should

    def _throttle(self):
        now = datetime.utcnow()
        oldest = self.timestamps[0]
        elapsed = (now - oldest).total_seconds()
        sleep_time = max(self.window_seconds - elapsed, 0)

        self.logger.warning(
            "Rate limit reached — throttling",
            extra={
                "current_requests": len(self.timestamps),
                "oldest_request_time": oldest.isoformat(),
                "elapsed_seconds": elapsed,
                "sleep_seconds": sleep_time,
            },
        )

        time.sleep(sleep_time + 0.01)

        self.logger.debug(
            "Throttle sleep complete",
            extra={"slept_seconds": sleep_time + 0.01},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        fn: Callable[[], Any],
        endpoint_name: str,
        metadata: Optional[dict] = None,
        max_retries: int = 5,
    ) -> Any:
        retry_count = 0
        metadata = metadata or {}

        self.logger.info(
            "Executing API call",
            extra={
                "endpoint": endpoint_name,
                "metadata": metadata,
                "max_retries": max_retries,
            },
        )

        while True:
            self.logger.debug(
                "Execution loop start",
                extra={
                    "endpoint": endpoint_name,
                    "retry_count": retry_count,
                    "queue_size": len(self.timestamps),
                },
            )

            if self._should_throttle():
                self._throttle()

            start_wall = time.time()
            start_utc = datetime.utcnow()

            self.logger.debug(
                "Issuing API request",
                extra={
                    "endpoint": endpoint_name,
                    "start_time_utc": start_utc.isoformat(),
                },
            )

            try:
                self.timestamps.append(start_utc)

                self.logger.debug(
                    "Timestamp appended",
                    extra={
                        "new_queue_size": len(self.timestamps),
                        "timestamps": [ts.isoformat() for ts in self.timestamps],
                    },
                )

                result = fn()

                duration_ms = int((time.time() - start_wall) * 1000)

                self.logger.info(
                    "API call success",
                    extra={
                        "endpoint": endpoint_name,
                        "duration_ms": duration_ms,
                        "retry_count": retry_count,
                        "queue_size": len(self.timestamps),
                        **metadata,
                    },
                )

                return result

            except Exception as e:
                import traceback

                self.logger.error(
                    f"API call failed: {endpoint_name}",
                    extra={
                        "retry_count": retry_count,
                        "exception": repr(e),
                        "traceback": traceback.format_exc(),
                    },
                )

                retry_count += 1
                backoff = min(2**retry_count, 30)

                self.logger.error(
                    "API call exception",
                    extra={
                        "endpoint": endpoint_name,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "retry_count": retry_count,
                        "max_retries": max_retries,
                        "backoff_seconds": backoff,
                        "queue_size": len(self.timestamps),
                        **metadata,
                    },
                )

                if retry_count >= max_retries:
                    self.logger.error(
                        "Max retries exceeded — raising exception",
                        extra={
                            "endpoint": endpoint_name,
                            "retry_count": retry_count,
                        },
                    )
                    raise

                self.logger.warning(
                    "Retrying API call after backoff",
                    extra={
                        "endpoint": endpoint_name,
                        "retry_count": retry_count,
                        "sleep_seconds": backoff,
                    },
                )

                time.sleep(backoff)
