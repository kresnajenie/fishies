from fastapi import FastAPI
import redis_test as re
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Define a list of origins that should be allowed to make requests
# You can use ["*"] to allow all origins
origins = [
    "https://fishies.techkyra.com",
    "https://kresnajenie.github.io",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,  # Allow credentials (cookies, headers)
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


@app.get("/")
async def read_root():
    return {"message": "Hello World!"}

@app.get("/getdata")
async def fetch_data(col: str):
# async def fetch_json():
    return {"data": re.retrieve_from_redis(col)}
