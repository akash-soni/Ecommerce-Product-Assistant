
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from prod_assistant.workflow.agentic_workflow_with_mcp_websearch import AgenticRAG


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # Initialize AgenticRAG once at startup
# rag_agent = AgenticRAG()


# @app.on_event("startup")
# async def startup_event():
#     """Load MCP tools asynchronously when FastAPI starts."""
#     await rag_agent.init_tools()

# ---------- FastAPI Endpoints ----------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/get", response_class=HTMLResponse)
async def chat(msg: str = Form(...)):
    """Call the Agentic RAG workflow."""
    rag_agent = AgenticRAG()
    answer = await rag_agent.run(msg)   # run() already returns final answer string
    print(f"Agentic Response: {answer}")
    return answer


# @app.post("/get", response_class=HTMLResponse)
# async def chat(msg: str = Form(...)):
#     """Call the Agentic RAG workflow."""
#     # Await the async run() method
#     answer = await rag_agent.run(msg)
#     print(f"Agentic Response: {answer}")
#     return answer