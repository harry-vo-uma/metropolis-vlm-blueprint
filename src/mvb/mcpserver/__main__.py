"""`python -m mvb.mcpserver` -> stdio MCP server."""
from .server import serve_stdio

if __name__ == "__main__":
    serve_stdio()
