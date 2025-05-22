import os
from typing import List, Dict
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# Load env vars
load_dotenv()

# Load multiple service accounts
SERVICE_ACCOUNTS = {
    "default": os.getenv("FIREBASE_CRED_JSON"),
    "project1": os.getenv("FIREBASE_PROJECT1_JSON"),
    #tambah lagi disini
}

firebase_apps = {}

for name, path in SERVICE_ACCOUNTS.items():
    if name not in firebase_admin._apps and path:
        cred = credentials.Certificate(path)
        firebase_apps[name] = firebase_admin.initialize_app(cred, name=name)

def get_firestore(project: str = "default"):
    if project not in firebase_apps:
        raise ValueError(f"Project '{project}' is not initialized.")
    return firestore.client(app=firebase_apps[project])

# MCP Server
mcp = FastMCP("firebase-files", port=8003)

@mcp.tool()
def list_collections(project: str = "default") -> List[str]:
    try:
        db = get_firestore(project)
        return [c.id for c in db.collections()]
    except Exception as e:
        return [f"Error: {str(e)}"]

@mcp.tool()
def list_files(collection_name: str, folder: str = "", project: str = "default") -> List[Dict]:
    try:
        db = get_firestore(project)
        query = db.collection(collection_name)
        if folder:
            query = query.where("folder", "==", folder)
        docs = query.stream()
        return [doc.to_dict() | {"id": doc.id} for doc in docs]
    except Exception as e:
        return [{"error": str(e)}]

@mcp.tool()
def get_file(collection_name: str, file_id: str, project: str = "default") -> Dict:
    try:
        db = get_firestore(project)
        doc = db.collection(collection_name).document(file_id).get()
        if doc.exists:
            return doc.to_dict() | {"id": doc.id}
        return {"error": f"File with ID {file_id} not found in {collection_name}."}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def update_file_metadata(collection_name: str, file_id: str, metadata: Dict, project: str = "default") -> str:
    try:
        db = get_firestore(project)
        ref = db.collection(collection_name).document(file_id)
        if not ref.get().exists:
            return f"File with ID {file_id} not found in {collection_name}."
        ref.update(metadata)
        return "Metadata updated successfully."
    except Exception as e:
        return f"Failed to update metadata: {str(e)}"

@mcp.tool()
def create_file(
    collection_name: str,
    file_id: str,
    fileName: str = "",
    folder: str = "",
    owner: str = "",
    createdBy: str = "",
    createdAt: str = "",
    project: str = "default"
) -> str:
    try:
        db = get_firestore(project)
        if not createdAt:
            createdAt = datetime.utcnow().strftime("%Y-%m-%d")

        doc_ref = db.collection(collection_name).document(file_id)
        if doc_ref.get().exists:
            return f"File with ID {file_id} already exists in {collection_name}."

        docs = db.collection(collection_name).limit(1).stream()
        template = {}
        for doc in docs:
            template = doc.to_dict()
            break

        if not template:
            template = {
                "id": file_id,
                "fileName": fileName,
                "folder": folder,
                "owner": owner,
                "createdAt": createdAt
            }
        else:
            for key in template:
                if key == "id":
                    template[key] = file_id
                elif key == "fileName":
                    template[key] = fileName
                elif key in ["folder", "owner", "createdBy"]:
                    template[key] = locals().get(key, "")
                elif key == "createdAt":
                    template[key] = createdAt
                else:
                    template[key] = template.get(key, "")

        doc_ref.set(template)
        return f"File '{fileName}' created successfully in collection '{collection_name}'."
    except Exception as e:
        return f"Failed to create file: {str(e)}"

@mcp.resource("firebasefiles://{project}/{collection_name}/{file_id}")
def get_file_resource(project: str, collection_name: str, file_id: str) -> str:
    data = get_file(collection_name, file_id, project)
    if "error" in data:
        return f"# Error\n\n{data['error']}"
    content = f"# File: {data.get('fileName', 'Unknown')}\n\n"
    for key, value in data.items():
        if key != "id":
            content += f"- **{key}**: {value}\n"
    return content

@mcp.resource("firebasefiles://{project}/{collection_name}/list")
def get_all_files_resource(project: str, collection_name: str) -> str:
    files = list_files(collection_name, project=project)
    if files and "error" in files[0]:
        return f"# Error\n\n{files[0]['error']}"
    content = f"# Firebase Files in `{collection_name}` ({project})\n\n"
    for f in files[:20]:
        content += f"## {f.get('fileName', 'Unknown')}\n"
        content += f"- **File ID**: {f.get('id')}\n"
        content += f"- **Folder**: {f.get('folder', '-')}\n"
        content += f"- **Created At**: {f.get('createdAt', '-')}\n"
        content += f"- **Owner**: {f.get('owner', '-')}\n"
        content += "---\n"
    return content

@mcp.prompt()
def generate_file_insight_prompt(project: str, collection_name: str, file_id: str) -> str:
    return f"""
    Use the `get_file` tool to retrieve the file with ID `{file_id}` from the collection `{collection_name}` in project `{project}`.

    Then provide:
    - Summary of the file metadata
    - Explanation of any important fields
    - Suggestions for updates or review actions
    """

if __name__ == "__main__":
    mcp.run(transport="sse")
