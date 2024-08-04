from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from langserve import add_routes

app = FastAPI()

from main_graph import graph

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

add_routes(
   app,
   graph.with_types(input_type=dict, output_type=dict),
   path="/chat",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)