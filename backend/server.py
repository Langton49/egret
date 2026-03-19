import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from data_router import router as aoi_router
from habitat_router import router as habitat_router
from research_router import router as research_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialise RAG service on startup (builds/reloads ChromaDB if needed)
    try:
        from rag_service import rag
        rag.init()
        logger.info("RAG service ready")
    except Exception as e:
        logger.warning("RAG service init failed (search/grant unavailable): %s", e)
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"message": "OK"}


app.include_router(aoi_router)
app.include_router(habitat_router)
app.include_router(research_router, prefix="/api/research")

