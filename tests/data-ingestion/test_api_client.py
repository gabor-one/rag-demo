# This file is mostly AI generated. Very few lines were manually edited.
# Not super interesting.

import asyncio
import pytest
from unittest.mock import AsyncMock, patch
import aiohttp

from rag_solution.data_ingestion.api_client import APIClient
from rag_solution.data_ingestion.ingestion_config import IngestionConfig
from rag_solution.data_ingestion.model import DocumentIngest, DocumentsIngestRequest


@pytest.fixture
def config():
    """Create a test configuration."""
    return IngestionConfig(
        api_endpoint="http://test.example.com/ingest",
        api_timeout=30,
        max_retries=3,
        retry_delay=1
    )


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        DocumentIngest(text="Test document 1", metadata={"source": "test1.txt"}),
        DocumentIngest(text="Test document 2", metadata={"source": "test2.txt"}),
    ]


class TestAPIClient:
    """Test cases for APIClient class."""

    @pytest.mark.asyncio
    async def test_context_manager_initialization(self, config):
        """Test that APIClient properly initializes and closes session."""
        async with APIClient(config) as client:
            assert client.session is not None
            assert isinstance(client.session, aiohttp.ClientSession)
            assert client.session.timeout.total == config.api_timeout

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self, config):
        """Test that APIClient properly cleans up session on exit."""
        client = APIClient(config)
        
        async with client:
            session = client.session
            assert session is not None
        
        # Session should be closed after exiting context
        assert client.session is not None  # Reference still exists
        assert session.closed  # But the session itself is closed

    @pytest.mark.asyncio
    async def test_send_documents_success(self, config, sample_documents):
        """Test successful document sending."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.__aenter__.return_value = mock_response
            mock_response.__aexit__.return_value = None
            mock_post.return_value = mock_response

            async with APIClient(config) as client:
                result = await client.send_documents(sample_documents)
                
                assert result is True
                mock_post.assert_called_once()
                
                # Verify the request data
                call_args = mock_post.call_args
                assert call_args[1]['json'] == DocumentsIngestRequest(documents=sample_documents).model_dump()
                assert call_args[1]['headers'] == {"Content-Type": "application/json"}

    @pytest.mark.asyncio
    async def test_send_documents_without_session(self, config, sample_documents):
        """Test that sending documents without session raises error."""
        client = APIClient(config)
        
        with pytest.raises(RuntimeError, match="APIClient not properly initialized"):
            await client.send_documents(sample_documents)

    @pytest.mark.asyncio
    async def test_send_documents_rate_limit_with_retry_after_header(self, config, sample_documents):
        """Test rate limiting with Retry-After header."""
        with patch('aiohttp.ClientSession.post') as mock_post, \
             patch('asyncio.sleep') as mock_sleep:
            
            # First call returns 429, second call succeeds
            mock_response_429 = AsyncMock()
            mock_response_429.status = 429
            mock_response_429.headers = {"Retry-After": "5"}
            mock_response_429.__aenter__.return_value = mock_response_429
            mock_response_429.__aexit__.return_value = None
            
            mock_response_200 = AsyncMock()
            mock_response_200.status = 200
            mock_response_200.__aenter__.return_value = mock_response_200
            mock_response_200.__aexit__.return_value = None
            
            mock_post.side_effect = [mock_response_429, mock_response_200]

            async with APIClient(config) as client:
                result = await client.send_documents(sample_documents)
                
                assert result is True
                assert mock_post.call_count == 2
                mock_sleep.assert_called_once_with(5)  # Should use Retry-After value

    @pytest.mark.asyncio
    async def test_send_documents_rate_limit_without_retry_after_header(self, config, sample_documents):
        """Test rate limiting without Retry-After header (exponential backoff)."""
        with patch('aiohttp.ClientSession.post') as mock_post, \
             patch('asyncio.sleep') as mock_sleep:
            
            # First call returns 429, second call succeeds
            mock_response_429 = AsyncMock()
            mock_response_429.status = 429
            mock_response_429.headers = {}  # No Retry-After header
            mock_response_429.__aenter__.return_value = mock_response_429
            mock_response_429.__aexit__.return_value = None
            
            mock_response_200 = AsyncMock()
            mock_response_200.status = 200
            mock_response_200.__aenter__.return_value = mock_response_200
            mock_response_200.__aexit__.return_value = None
            
            mock_post.side_effect = [mock_response_429, mock_response_200]

            async with APIClient(config) as client:
                result = await client.send_documents(sample_documents)
                
                assert result is True
                assert mock_post.call_count == 2
                # Should use exponential backoff: (2^0) * retry_delay = 1 * 1 = 1
                mock_sleep.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_send_documents_api_error_with_retry(self, config, sample_documents):
        """Test API error with retry logic."""
        with patch('aiohttp.ClientSession.post') as mock_post, \
             patch('asyncio.sleep') as mock_sleep:
            
            # First call returns 500, second call succeeds
            mock_response_500 = AsyncMock()
            mock_response_500.status = 500
            mock_response_500.text.return_value = "Internal Server Error"
            mock_response_500.__aenter__.return_value = mock_response_500
            mock_response_500.__aexit__.return_value = None
            
            mock_response_200 = AsyncMock()
            mock_response_200.status = 200
            mock_response_200.__aenter__.return_value = mock_response_200
            mock_response_200.__aexit__.return_value = None
            
            mock_post.side_effect = [mock_response_500, mock_response_200]

            async with APIClient(config) as client:
                result = await client.send_documents(sample_documents)
                
                assert result is True
                assert mock_post.call_count == 2
                mock_sleep.assert_called_once_with(1)  # (2^0) * 1 = 1

    @pytest.mark.asyncio
    async def test_send_documents_max_retries_exceeded(self, config, sample_documents):
        """Test max retries exceeded scenario."""
        with patch('aiohttp.ClientSession.post') as mock_post, \
             patch('asyncio.sleep') as mock_sleep:
            
            # Always return 500
            mock_response_500 = AsyncMock()
            mock_response_500.status = 500
            mock_response_500.text.return_value = "Internal Server Error"
            mock_response_500.__aenter__.return_value = mock_response_500
            mock_response_500.__aexit__.return_value = None
            
            mock_post.return_value = mock_response_500

            async with APIClient(config) as client:
                result = await client.send_documents(sample_documents)
                
                assert result is False
                # Should try: initial + max_retries = 1 + 3 = 4 times
                assert mock_post.call_count == 4
                # Should sleep 3 times (not on the last failed attempt)
                assert mock_sleep.call_count == 3

    @pytest.mark.asyncio
    async def test_send_documents_timeout_error(self, config, sample_documents):
        """Test timeout error handling."""
        with patch('aiohttp.ClientSession.post') as mock_post, \
             patch('asyncio.sleep') as mock_sleep:
            
            # First call times out, second call succeeds
            mock_post.side_effect = [
                asyncio.TimeoutError("Request timed out"),
                AsyncMock(__aenter__=AsyncMock(return_value=AsyncMock(status=200)),
                         __aexit__=AsyncMock(return_value=None))
            ]

            async with APIClient(config) as client:
                result = await client.send_documents(sample_documents)
                
                assert result is True
                assert mock_post.call_count == 2
                mock_sleep.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_send_documents_generic_exception(self, config, sample_documents):
        """Test generic exception handling."""
        with patch('aiohttp.ClientSession.post') as mock_post, \
             patch('asyncio.sleep') as mock_sleep:
            
            # First call raises exception, second call succeeds
            mock_post.side_effect = [
                Exception("Connection error"),
                AsyncMock(__aenter__=AsyncMock(return_value=AsyncMock(status=200)),
                         __aexit__=AsyncMock(return_value=None))
            ]

            async with APIClient(config) as client:
                result = await client.send_documents(sample_documents)
                
                assert result is True
                assert mock_post.call_count == 2
                mock_sleep.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_send_documents_timeout_max_retries_exceeded(self, config, sample_documents):
        """Test timeout with max retries exceeded."""
        with patch('aiohttp.ClientSession.post') as mock_post, \
             patch('asyncio.sleep') as mock_sleep:
            
            # Always timeout
            mock_post.side_effect = asyncio.TimeoutError("Request timed out")

            async with APIClient(config) as client:
                result = await client.send_documents(sample_documents)
                
                assert result is False
                # Should try: initial + max_retries = 1 + 3 = 4 times
                assert mock_post.call_count == 4
                # Should sleep 3 times (not on the last failed attempt)
                assert mock_sleep.call_count == 3

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self, config, sample_documents):
        """Test that exponential backoff calculates delays correctly."""
        with patch('aiohttp.ClientSession.post') as mock_post, \
             patch('asyncio.sleep') as mock_sleep:
            
            # Always return 500 to trigger retries
            mock_response_500 = AsyncMock()
            mock_response_500.status = 500
            mock_response_500.text.return_value = "Internal Server Error"
            mock_response_500.__aenter__.return_value = mock_response_500
            mock_response_500.__aexit__.return_value = None
            
            mock_post.return_value = mock_response_500

            async with APIClient(config) as client:
                result = await client.send_documents(sample_documents)
                
                assert result is False
                
                # Check exponential backoff delays
                expected_delays = [
                    (2**0) * config.retry_delay,  # 1 * 1 = 1
                    (2**1) * config.retry_delay,  # 2 * 1 = 2
                    (2**2) * config.retry_delay,  # 4 * 1 = 4
                ]
                
                sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
                assert sleep_calls == expected_delays

    @pytest.mark.asyncio
    async def test_send_empty_documents_list(self, config):
        """Test sending empty documents list."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.__aenter__.return_value = mock_response
            mock_response.__aexit__.return_value = None
            mock_post.return_value = mock_response

            async with APIClient(config) as client:
                result = await client.send_documents([])
                
                assert result is True
                mock_post.assert_not_called()  # No post call should be made for empty list

    @pytest.mark.asyncio
    async def test_config_properties_used_correctly(self, sample_documents):
        """Test that config properties are used correctly."""
        custom_config = IngestionConfig(
            api_endpoint="http://custom.example.com/api",
            api_timeout=120,
            max_retries=2,
            retry_delay=5
        )
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.__aenter__.return_value = mock_response
            mock_response.__aexit__.return_value = None
            mock_post.return_value = mock_response

            async with APIClient(custom_config) as client:
                await client.send_documents(sample_documents)
                
                # Check correct endpoint was used
                mock_post.assert_called_once()
                post_call_args = mock_post.call_args
                assert post_call_args[0][0] == "http://custom.example.com/api"
                
                # Verify the request data and headers
                assert post_call_args[1]['json'] == DocumentsIngestRequest(documents=sample_documents).model_dump()
                assert post_call_args[1]['headers'] == {"Content-Type": "application/json"}
