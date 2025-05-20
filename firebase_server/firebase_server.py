import os
import json
from typing import List, Dict
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import firebase_admin
from firebase_admin import credentials, firestore

# Load env vars
load_dotenv()

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(os.getenv("FIREBASE_CRED_JSON"))
    firebase_admin.initialize_app(cred)

db = firestore.client()

# MCP Server
mcp = FastMCP("firebase-files", port=8003)

@mcp.tool()
def list_files(folder: str = "") -> List[Dict]:
    try:
        query = db.collection("files")
        if folder:
            query = query.where("folder", "==", folder)
        docs = query.stream()
        return [doc.to_dict() | {"id": doc.id} for doc in docs]
    except Exception as e:
        return [{"error": str(e)}]

@mcp.tool()
def get_file(file_id: str) -> Dict:
    try:
        doc = db.collection("files").document(file_id).get()
        if doc.exists:
            return doc.to_dict() | {"id": doc.id}
        return {"error": f"File with ID {file_id} not found."}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def update_file_metadata(file_id: str, metadata: Dict) -> str:
    try:
        ref = db.collection("files").document(file_id)
        if not ref.get().exists:
            return f"File with ID {file_id} not found."
        ref.update(metadata)
        return "Metadata updated successfully."
    except Exception as e:
        return f"Failed to update metadata: {str(e)}"

@mcp.resource("firebasefiles://{file_id}")
def get_file_resource(file_id: str) -> str:
    data = get_file(file_id)
    if "error" in data:
        return f"# Error\n\n{data['error']}"

    content = f"# File: {data['fileName']}\n\n"
    for key, value in data.items():
        if key != "id":
            content += f"- **{key}**: {value}\n"
    return content

@mcp.resource("firebasefiles://list")
def get_all_files_resource() -> str:
    files = list_files()
    if files and "error" in files[0]:
        return f"# Error\n\n{files[0]['error']}"

    content = "# Firebase Files\n\n"
    for f in files[:20]:
        content += f"## {f.get('fileName', 'Unknown')}\n"
        content += f"- **File ID**: {f.get('id')}\n"
        content += f"- **Folder**: {f.get('folder', '-')}\n"
        content += f"- **Created At**: {f.get('createdAt', '-')}\n"
        content += f"- **Owner**: {f.get('owner', '-')}\n"
        content += "---\n"
    return content

@mcp.prompt()
def generate_file_insight_prompt(file_id: str) -> str:
    return f"""
    Use the `get_file` tool to retrieve the file with ID `{file_id}` from Firebase.

    Then provide:
    - Summary of the file metadata
    - Explanation of any important fields
    - Suggestions for updates or review actions
    """

if __name__ == "__main__":
    mcp.run(transport="sse")