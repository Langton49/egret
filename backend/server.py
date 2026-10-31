import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from habitat_router import router as habitat_router

logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"message": "OK"}


app.include_router(habitat_router)
