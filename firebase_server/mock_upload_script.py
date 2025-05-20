import os
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Firebase setup
cred_path = os.getenv("FIREBASE_CRED_JSON")
firebase_admin.initialize_app(credentials.Certificate(cred_path))
db = firestore.client()

# Mock data
files = [
    {
        "id": "123",
        "fileName": "TestFile1.pdf",
        "folder": "Reports",
        "createdAt": "2024-05-19",
        "owner": "tester"
    },
    {
        "id": "456",
        "fileName": "TestFile2.docx",
        "folder": "Invoices",
        "createdAt": "2024-05-18",
        "owner": "tester"
    },
    {
        "id": "789",
        "fileName": "Specs2025.xlsx",
        "folder": "Engineering",
        "createdAt": "2025-01-10",
        "owner": "engineer1"
    }
]

# Upload to Firestore
for f in files:
    doc_ref = db.collection("files").document(f["id"])
    doc_ref.set(f)

print("Mock files uploaded.")