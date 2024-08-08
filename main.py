from api import app
import asyncio
import subprocess
import uvicorn


async def run_fastapi():
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    await server.serve()

async def run_streamlit():
    process = await asyncio.create_subprocess_exec(
        'streamlit', 'run', 'frontend.py',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        print(line.decode(), end='')

    await process.wait()

async def main():
    await asyncio.gather(
        run_fastapi(),
        run_streamlit()
    )

if __name__ == "__main__":
    asyncio.run(main())