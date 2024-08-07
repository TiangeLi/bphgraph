from fastapi import FastAPI
from fastapi.responses import StreamingResponse, RedirectResponse
from main_graph import graph
import json
from fastapi.middleware.cors import CORSMiddleware
from os import getenv
from dotenv import load_dotenv
load_dotenv()

origins = [
    getenv('FRONTEND_URL'),
]

async def run_graph(input: dict):
    curr_node = ''
    last_node = ''
    content_to_stream = ''
    async for output in graph.astream_log(input, include_types=['llm']):
        for op in output.ops:
            if isinstance(op['value'], dict) and 'metadata' in op['value'] and 'langgraph_node' in op['value']['metadata']:
                curr_node = op['value']['metadata']['langgraph_node']
            elif op["path"][:6] == "/logs/" and op["path"][-18:] == "/streamed_output/-" and curr_node == 'chat':
                content_to_stream = op["value"].content
            elif op["path"] == "/streamed_output/-":
                if curr_node == 'summarizer':
                    content_to_stream = op['value']['summarizer']['sources']
                elif curr_node == 'filter':
                    try:
                        content_to_stream = [{'page_content': c.page_content, 'metadata': c.metadata} for c in op['value']['filter']['filtered_retrieved']]
                    except:
                        curr_node = ''
            if curr_node != last_node:
                yield json.dumps({'node': curr_node})+'\n'
                last_node = curr_node
            if content_to_stream:
                yield json.dumps({'content': content_to_stream})+'\n'
                content_to_stream = ''

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"]
)

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

@app.post("/chat")
async def read_item(input: dict):
    return StreamingResponse(run_graph(input))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)