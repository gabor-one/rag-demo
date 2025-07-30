from loguru import logger
from typing import List, Optional
from ingestion_config import IngestionConfig
from rag_solution.data_ingestion.model import DocumentIngest, DocumentsIngestRequest
import aiohttp
import asyncio


class APIClient:
    """Handles API communication with retry logic."""

    def __init__(self, config: IngestionConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=self.config.api_timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def send_documents(self, documents: List[DocumentIngest]) -> bool:
        """Send documents to the API endpoint with retry logic."""
        if not self.session:
            raise RuntimeError("APIClient not properly initialized")

        request_data = DocumentsIngestRequest(documents=documents)

        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.session.post(
                    self.config.api_endpoint,
                    json=request_data.model_dump(),
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        logger.debug(f"Successfully sent {len(documents)} documents")
                        return True
                    elif response.status == 429:
                        # Rate limit - retry with exponential backoff
                        # Let the server tell us when to retry
                        # SlowApi fills this header on 429
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            delay = int(retry_after)
                        else:
                            delay = (2**attempt) * self.config.retry_delay

                        logger.warning(
                            f"Rate limited. Retrying after {delay} seconds..."
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        response_text = await response.text()
                        logger.error(f"API error {response.status}: {response_text}")

                        if attempt < self.config.max_retries:
                            delay = (2**attempt) * self.config.retry_delay
                            logger.info(
                                f"Retrying in {delay} seconds... (attempt {attempt + 1}/{self.config.max_retries})"
                            )
                            await asyncio.sleep(delay)
                        else:
                            logger.error("Max retries exceeded for batch")
                            return False

            except (asyncio.TimeoutError, Exception) as e:
                if e is asyncio.TimeoutError:
                    logger.error(f"Request timed out (attempt {attempt + 1}): {e}")
                else:
                    logger.error(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries:
                    delay = (2**attempt) * self.config.retry_delay
                    await asyncio.sleep(delay)
                else:
                    return False

        return False
