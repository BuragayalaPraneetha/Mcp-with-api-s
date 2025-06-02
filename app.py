import os
import requests
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from openai import OpenAI
from mcp.server.fastmcp import FastMCP
from apikey import qdrant_client

# ─── Load config ─────────────────────────────────────────
load_dotenv()
OPENAI_KEY    = os.getenv("OPENAI_API_KEY")
SCHOOL_URL    = os.getenv("SCHOOL_API_URL")
EMBED_MODEL   = "text-embedding-ada-002"
LLM_MODEL     = "gpt-3.5-turbo"

# ─── LLM & MCP setup ──────────────────────────────────────
llm = OpenAI(api_key=OPENAI_KEY)
mcp = FastMCP(name="MyMCP", llm_client=llm)

@mcp.tool()
def school_search_tool(area: str, city: str) -> dict:
    """Fetch school list from external API"""
    resp = requests.get(
        SCHOOL_URL,
        params={"area": area, "city": city},
        timeout=5
    )
    resp.raise_for_status()
    return resp.json()

@mcp.tool()
def summarize_schools_tool(area: str, city: str, abridged: bool = True) -> str:
    """Convert raw school JSON into user-friendly text"""
    raw = school_search_tool(area, city)
    instruction = (
        "Summarize the following school data in 2-3 sentences."
        if abridged else
        "Summarize the following school data in detail."
    )
    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user",   "content": str(raw)}
        ]
    )
    return resp.choices[0].message.content

@mcp.tool()
def vector_search_tool(query: str, top_k: int = 5) -> list:
    """Embed the query and fetch nearest vectors from Qdrant"""
    emb = llm.embeddings.create(model=EMBED_MODEL, input=query)
    vector = emb.data[0].embedding
    hits = qdrant_client.search(
        collection_name="default",
        query_vector=vector,
        limit=top_k
    )
    return hits

# ─── Flask app ────────────────────────────────────────────
app = Flask(__name__, static_folder="frontend", static_url_path="")

@app.route("/mcp", methods=["POST"])
def handle_mcp():
    return jsonify(mcp.run(request.get_json()))

@app.route("/chat", methods=["POST"])
def handle_chat():
    data = request.get_json()
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return jsonify({"response": resp.choices[0].message.content})

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)