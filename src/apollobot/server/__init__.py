"""ApolloBot MCP Server — exposes ApolloBot as an MCP tool provider."""

from apollobot.server.app import mcp


def create_server() -> "FastMCP":
    """Return the FastMCP server instance."""
    return mcp


def run_server() -> None:
    """Entry point for the apollobot-server script."""
    import sys

    transport = "stdio"
    host = "0.0.0.0"
    port = 8080

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--transport" and i + 1 < len(args):
            transport = args[i + 1]
            i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        elif args[i] == "--host" and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        else:
            i += 1

    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=transport, host=host, port=port)
