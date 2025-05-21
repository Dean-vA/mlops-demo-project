from fastapi import FastAPI

app = FastAPI(title="My FastAPI App")

# Define a simple endpoint
@app.get("/")
async def root():
    return {"message": "Hello World!"}

#add health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy"}
