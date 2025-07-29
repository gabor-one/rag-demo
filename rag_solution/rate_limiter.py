from fastapi import FastAPI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from rag_solution.settings import settings

limiter: Limiter | None = None


def setup_limiter(app: FastAPI):
    if settings.DEFAULT_RATE_LIMIT is None:
        return

    # Redis backend is needed if multiple app instances are used
    global limiter
    limiter = Limiter(key_func=get_remote_address, default_limits=[settings.DEFAULT_RATE_LIMIT])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
