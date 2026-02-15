from fastapi import FastAPI
from data_router import router as aoi_router
from habitat_router import router as habitat_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"message": "OK"}

app.include_router(aoi_router)
app.include_router(habitat_router)

