import os
import requests
import json
from typing import List, Dict
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

SAP_BASE_URL = os.getenv("SAP_BASE_URL")
SAP_USERNAME = os.getenv("SAP_USERNAME")
SAP_PASSWORD = os.getenv("SAP_PASSWORD")
AUTH = (SAP_USERNAME, SAP_PASSWORD)

mcp = FastMCP("sap-files", port=8010)


@mcp.tool()
def list_files(folder: str = "") -> List[Dict]:
    try:
        filter_query = f"?$filter=Folder eq '{folder}'" if folder else ""
        url = f"{SAP_BASE_URL}/Files{filter_query}"
        #res = requests.get(url, auth=AUTH)
        res = requests.get(url)
        res.raise_for_status()
        return res.json().get("d", {}).get("results", [])
    except Exception as e:
        return [{"error": str(e)}]


@mcp.tool()
def get_file(file_id: str) -> Dict:
    try:
        url = f"{SAP_BASE_URL}/Files('{file_id}')"
        res = requests.get(url, auth=AUTH)
        res.raise_for_status()
        return res.json().get("d", {})
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def update_file_metadata(file_id: str, metadata: Dict) -> str:
    try:
        url = f"{SAP_BASE_URL}/Files('{file_id}')"
        headers = {
            "Content-Type": "application/json",
            "X-Requested-With": "XMLHttpRequest",
            "x-csrf-token": "Fetch"
        }

        # Fetch CSRF token
        token_res = requests.get(url, auth=AUTH, headers=headers)
        csrf_token = token_res.headers.get("x-csrf-token")
        if not csrf_token:
            return "Failed to fetch CSRF token"

        headers["x-csrf-token"] = csrf_token
        res = requests.patch(url, auth=AUTH, headers=headers, json=metadata)
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

    content = "# SAP Files\n\n"
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
Use the `get_file` tool to retrieve the file with ID `{file_id}` from SAP.

Then provide:
- Summary of the file metadata
- Explanation of any important fields
- Suggestions for updates or review actions
"""


if __name__ == "__main__":
    mcp.run(transport="sse")