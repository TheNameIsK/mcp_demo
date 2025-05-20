import os
import requests
from typing import List, Dict
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load environment variables
load_dotenv()

# Load config from .env
SAP_BASE_URL = os.getenv("SAP_BASE_URL", "http://localhost:3000")

# Start MCP server
mcp = FastMCP("sap-local", port=8002)

@mcp.tool()
def list_files(folder: str = "") -> List[Dict]:
    try:
        url = f"{SAP_BASE_URL}/Files"
        res = requests.get(url)
        res.raise_for_status()
        files = res.json()

        if folder:
            files = [f for f in files if f.get("Folder", "").lower() == folder.lower()]

        return files
    except Exception as e:
        return [{"error": str(e)}]

@mcp.tool()
def get_file(file_id: str) -> Dict:
    try:
        url = f"{SAP_BASE_URL}/Files/{file_id}"
        res = requests.get(url)
        print("Requesting:", url)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def update_file_metadata(file_id: str, metadata: Dict) -> str:
    try:
        url = f"{SAP_BASE_URL}/Files/{file_id}"
        headers = {"Content-Type": "application/json"}
        res = requests.patch(url, headers=headers, json=metadata)
        res.raise_for_status()
        return "Metadata updated successfully"
    except Exception as e:
        return f"Failed to update metadata: {str(e)}"

@mcp.resource("sapfiles://{file_id}")
def get_file_resource(file_id: str) -> str:
    data = get_file(file_id)
    if "error" in data:
        return f"# Error\n\n{data['error']}"

    content = f"# File: {file_id}\n\n"
    for key, value in data.items():
        content += f"- **{key}**: {value}\n"
    return content

@mcp.resource("sapfiles://list")
def get_all_files_resource() -> str:
    files = list_files()
    if files and "error" in files[0]:
        return f"# Error\n\n{files[0]['error']}"

    content = "# SAP Files (Local)\n\n"
    for f in files[:20]:
        content += f"## {f.get('FileName', 'Unknown')}\n"
        content += f"- **File ID**: {f.get('FileID', 'N/A')}\n"
        content += f"- **Folder**: {f.get('Folder', '-')}\n"
        content += f"- **Created At**: {f.get('CreatedAt', '-')}\n"
        content += f"- **Owner**: {f.get('Owner', '-')}\n"
        content += "---\n"
    return content

@mcp.prompt()
def generate_file_insight_prompt(file_id: str) -> str:
    return f"""
    Use the `get_file` tool to retrieve the file with ID `{file_id}` from the local SAP mock server.

    Then provide:
    - Summary of the file metadata
    - Explanation of any important fields
    - Suggestions for updates or review actions
    """

if __name__ == "__main__":
    mcp.run(transport="sse")