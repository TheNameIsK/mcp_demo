
# Model Context Protocol (MCP) Integration System

This project aims to build a Model Context Protocol (MCP)-based system to allow seamless integration between AI agents and various enterprise data sources, such as SAP and Firebase. The system enables tools to be defined on data source servers and accessed via a unified MCP client interface—either through terminal or a Streamlit-based chatbot UI.

## Project Objectives

The primary goal is to create a standardized mechanism where AI models can query multiple back-end data systems using MCP, retrieve structured information, and optionally perform actions via tool calling.

## Current Progress

- Mock MCP Servers using SSE (Server-Sent Events):
  - SAP Server: Mock server for accessing SAP-style data.
  - Firebase Server: Integrated with Firestore to list files, fetch metadata, and update fields.
  
- Tool Implementation & Testing:
  - Tools are registered on each MCP server and verified to return correct data.
  
- MCP Client Chatbot:
  - Implemented a working MCP chatbot with support for:
    - Terminal interface.
    - Web interface using Streamlit.
  - Supports tool calling using OpenAI GPT-4o with tool_choice="auto".

- Verified Functionality:
  - Firebase server tools have been tested and are operating correctly.

## Features Pending Implementation

- SAP Server Testing:
  - Requires real SAP base URL, username, and password for integration.

- Extended Tooling:
  - Add tools for additional workflows and data queries.

- Authentication Mechanism:
  - Secure access to data sources using session-based or token-based authentication.

- AI System Integration:
  - Connect this MCP system to the company’s internal AI assistant for unified access.

- New Server Integration:
  - Build additional MCP servers for any new data sources.

## How to Run Locally

### 1. Clone the Repository

    git clone https://github.com/your-org/your-repo-name.git
    cd your-repo-name

### 2. Set Up Your Environment

Ensure Python 3.9+ is installed. Then create and activate a virtual environment:

    python -m venv .venv
    source .venv/bin/activate   # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt

Also, create a .env file with the required Firebase credentials path:

    FIREBASE_CRED_JSON=path/to/your/firebase/cred.json

### 3. Configure Your MCP Server Connections

Edit server_config.json with your server entries. Example:

    {
      "mcpServers": {
        "firebase": {
          "url": "http://localhost:8003"
        },
        "sap": {
          "url": "http://localhost:8002"
        }
      }
    }

Make sure the url corresponds to the correct MCP server address and port.

### 4. Run the MCP Server(s)

To start the Firebase-based MCP server:

    python firebase_server.py

Do the same for other servers, e.g., SAP, if implemented.

### 5. Launch the Chatbot (Streamlit UI)

Run the chatbot interface:

    streamlit run mcp_chatbot.py

Ask questions via the UI and let the OpenAI model invoke tools registered on your MCP servers.

## Folder Structure (Optional)

    ├── firebase_server.py         # Firebase-backed MCP server
    ├── sap_server.py              # (Optional) SAP mock MCP server
    ├── mcp_chatbot.py             # Client chatbot with Streamlit UI
    ├── server_config.json         # MCP server connection configuration
    ├── .env                       # Firebase credential reference
    ├── requirements.txt           # Dependencies
    └── README.txt                 # This file

## Contact

For internal AI integration or technical support, please reach out to the AI Engineering team.
