"""
Docs Checker Worker

Subscribes to a Redis PubSub channel, downloads documents from public Google Drive
links (.pdf / .docx only), extracts text, and submits each document to the DeepSeek
AI checker. The raw AI response is published back on a separate Redis channel.

Message format consumed (JSON):
    {"job_id": <int>, "link": <str>}

Message format published (JSON):
    {
      "job_id":          <int>,
      "status":          0 | 1,
      "file_names":      []<str>,
      "result":          {"likelihood_score": int, "reasoning": str, "is_ai_generated": bool},
      "raw_ai_response": <dict>   # full ChatCompletion.model_dump()
    }

Run:
    python -m workers.docs_checker_worker
    make run-worker
"""

import asyncio
import json
import logging
import os
import shutil
import signal
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import redis.asyncio as aioredis

from utils.gdrive_downloader import GDriveDownloader
from utils.text_extractor import TextExtractor
from workers.ai_checker import AIChecker

load_dotenv()

# ---------------------------------------------------------------------------
# Config (read once at import; override via environment variables)
# ---------------------------------------------------------------------------

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
REDIS_INPUT_CHANNEL = os.environ.get("DOCS_CHECKER_INPUT_CHANNEL", "docs_checker:jobs")
REDIS_OUTPUT_CHANNEL = os.environ.get("DOCS_CHECKER_OUTPUT_CHANNEL", "docs_checker:results")
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "3"))

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
_raw_base = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_BASE_URL = _raw_base if _raw_base.endswith("/v1") else f"{_raw_base}/v1"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("docs_checker_worker")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class DocsCheckerWorker:
    """
    Orchestrates the full pipeline for each incoming job:
      Redis PubSub → GDriveDownloader → TextExtractor → AIChecker → Redis publish
    """

    def __init__(
        self,
        redis_url: str = REDIS_URL,
        input_channel: str = REDIS_INPUT_CHANNEL,
        output_channel: str = REDIS_OUTPUT_CHANNEL,
        max_concurrent: int = MAX_CONCURRENT_JOBS,
    ) -> None:
        self._redis_url = redis_url
        self._input_channel = input_channel
        self._output_channel = output_channel
        self._semaphore = asyncio.Semaphore(max_concurrent)

        self._downloader = GDriveDownloader()
        self._extractor = TextExtractor()
        self._ai_checker = AIChecker(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        logger.info("Docs Checker Worker starting up")
        logger.info("  Input channel  : %s", self._input_channel)
        logger.info("  Output channel : %s", self._output_channel)
        logger.info("  Max concurrency: %d", self._semaphore._value)
        logger.info("  DeepSeek base  : %s", DEEPSEEK_BASE_URL)

        redis_client = aioredis.from_url(self._redis_url, decode_responses=False)
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(self._input_channel)
        logger.info("Subscribed — waiting for messages...")

        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda: (logger.info("Shutdown signal received"), stop_event.set()),
            )

        try:
            while not stop_event.is_set():
                try:
                    message = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=True),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                if message and message.get("type") == "message" and message.get("data"):
                    asyncio.create_task(self._process_job(redis_client, message["data"]))
        finally:
            logger.info("Shutting down gracefully...")
            await pubsub.unsubscribe(self._input_channel)
            await pubsub.aclose()
            await redis_client.aclose()
            logger.info("Worker stopped.")

    # ------------------------------------------------------------------
    # Job pipeline
    # ------------------------------------------------------------------

    async def _process_job(self, redis_client: aioredis.Redis, raw_data: bytes) -> None:
        """Parse one PubSub message and run the full pipeline."""
        try:
            payload = json.loads(raw_data)
            job_id = payload["job_id"]
            link: str = payload["link"].strip()
        except (json.JSONDecodeError, KeyError, AttributeError) as exc:
            logger.error("Malformed message — skipping: %s", exc)
            return

        async with self._semaphore:
            logger.info("[job=%s] Starting — link: %s", job_id, link)
            tmp_dir = tempfile.mkdtemp(prefix=f"docs_checker_{job_id}_")
            try:
                await self._run_pipeline(redis_client, job_id, link, tmp_dir)
            except Exception as exc:
                logger.exception("[job=%s] Unexpected error: %s", job_id, exc)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    async def _run_pipeline(
        self,
        redis_client: aioredis.Redis,
        job_id: int,
        link: str,
        tmp_dir: str,
    ) -> None:
        # Step 1: Download files from Google Drive
        try:
            files = await asyncio.to_thread(self._downloader.download, link, tmp_dir)
        except PermissionError as exc:
            logger.warning("[job=%s] Skipped — link not public: %s", job_id, exc)
            return

        if not files:
            logger.warning("[job=%s] Skipped — no .pdf or .docx files found at link", job_id)
            return

        logger.info(
            "[job=%s] Found %d supported file(s): %s",
            job_id,
            len(files),
            [Path(f).name for f in files],
        )

        # Step 2: Check every file, collecting results before publishing
        per_file_results: list[dict] = []
        for file_path in files:
            file_name = Path(file_path).name
            result = await self._check_single_file(job_id, file_path, file_name)

            # Log token usage per file then strip the raw response — it must not
            # end up in the aggregated message published to Redis.
            if "raw_ai_response" in result:
                await asyncio.to_thread(
                    self._log_token_usage, job_id, file_name, result["raw_ai_response"]
                )
                del result["raw_ai_response"]

            per_file_results.append(result)

        # Step 3: Aggregate across all files and publish a single message
        final_msg = self._aggregate_results(job_id, per_file_results)
        await redis_client.publish(self._output_channel, json.dumps(final_msg))
        logger.info(
            "[job=%s] Aggregated result published to '%s' (files=%d, avg_likelihood=%s)",
            job_id,
            self._output_channel,
            len(files),
            final_msg.get("result", {}).get("likelihood_score"),
        )

    async def _check_single_file(
        self, job_id: int, file_path: str, file_name: str
    ) -> dict:
        logger.info("[job=%s] Checking '%s'", job_id, file_name)
        try:
            text = await asyncio.to_thread(self._extractor.extract, file_path)
            if not text.strip():
                raise ValueError(f"No text could be extracted from '{file_name}'")

            ai_result = await self._ai_checker.check(text)
            parsed = ai_result["parsed"]
            logger.info(
                "[job=%s] '%s' → likelihood=%s, is_ai_generated=%s",
                job_id,
                file_name,
                parsed.get("likelihood_score"),
                parsed.get("is_ai_generated"),
            )
            return {
                "job_id": job_id,
                "status": 1,
                "file_name": file_name,
                "result": parsed,
                "raw_ai_response": ai_result["raw"],
            }
        except Exception as exc:
            logger.error("[job=%s] Failed to process '%s': %s", job_id, file_name, exc)
            return {
                "job_id": job_id,
                "status": 0,
                "file_name": file_name,
                "error": str(exc),
            }


    @staticmethod
    def _aggregate_results(job_id: int, results: list[dict]) -> dict:
        """
        Merge per-file results into a single message.

        - likelihood_score : average of all successful scores (rounded)
        - is_ai_generated  : true when that average exceeds 65
        - reasoning        : per-file reasonings labelled with the file name
        - file_names        : list of every file that was attempted
        - status           : 1 if at least one file succeeded, 0 if all failed
        """
        file_names = [r["file_name"] for r in results]
        successful = [r for r in results if r.get("status") == 1]

        if not successful:
            # Every file failed — surface the first error so the caller knows why
            first_error = results[0] if results else {}
            return {
                "job_id": job_id,
                "status": 0,
                "file_name": file_names,
                "error": first_error.get("error", "All files failed to process"),
            }

        scores = [r["result"]["likelihood_score"] for r in successful]
        avg_score = round(sum(scores) / len(scores))

        reasoning = "\n\n".join(
            f"[{r['file_name']}] {r['result'].get('reasoning', '')}"
            for r in successful
        )

        logger.info(
            "[job=%s] Scores per file: %s → average: %d",
            job_id,
            {r["file_name"]: r["result"]["likelihood_score"] for r in successful},
            avg_score,
        )

        return {
            "job_id": job_id,
            "status": 1,
            "file_names": file_names,
            "result": {
                "likelihood_score": avg_score,
                "reasoning": reasoning,
                "is_ai_generated": avg_score > 65,
            },
        }

    @staticmethod
    def _log_token_usage(job_id: int, file_name: str, raw_response: dict) -> None:
        """Write token usage for one AI call to ./logs/docs-checker/<timestamp>.json."""
        usage = raw_response.get("usage") or {}
        record = {
            "token_usage": {
                "extractor": "docs_checker",
                "job_id": job_id,
                "file_name": file_name,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "is_estimated": False,
            }
        }

        log_dir = Path("logs/docs-checker")
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = log_dir / f"docs_checker_token_usage_{timestamp}.json"
        log_path.write_text(json.dumps(record, indent=2))

        logger.info(
            "[job=%s] '%s' token usage — prompt=%d, completion=%d, total=%d",
            job_id,
            file_name,
            record["token_usage"]["prompt_tokens"],
            record["token_usage"]["completion_tokens"],
            record["token_usage"]["total_tokens"],
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not DEEPSEEK_API_KEY:
        logger.error("DEEPSEEK_API_KEY is not set — aborting.")
        sys.exit(1)
    asyncio.run(DocsCheckerWorker().run())


if __name__ == "__main__":
    main()
