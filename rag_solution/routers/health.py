from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(tags=["Health"], include_in_schema=False)


@router.get("/health")
async def health():
    return JSONResponse(content={"status": "ok"})