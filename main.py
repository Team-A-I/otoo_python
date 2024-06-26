from fastapi import FastAPI

app = FastAPI()

@app.get("/fastapi-endpoint")
def read_fastapi():
    return {"message": "Hello from FastAPI"}
