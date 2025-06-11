# MCP Firebase & Supabase Analysis System

A powerful Model Context Protocol (MCP) system that provides intelligent data analysis capabilities for both Firebase Firestore and Supabase PostgreSQL databases. This system combines a unified MCP server with an intelligent chatbot client that leverages conversation history and context memory for efficient database operations.

## 🚀 Features

### Firebase Firestore Integration
- **Collection Management**: List and analyze Firebase collections across multiple projects
- **Advanced Analytics**: Comprehensive collection statistics, field analysis, and data type distribution
- **File Pattern Analysis**: Analyze file naming patterns, folder structures, and temporal trends
- **Smart Search & Filtering**: Advanced search capabilities with multiple filter options
- **Time-based Comparisons**: Compare data across different time periods
- **Collaboration Analysis**: Analyze user contributions and collaboration patterns

### Supabase PostgreSQL Integration
- **Schema Awareness**: Automatic database schema loading and analysis
- **Safe Query Execution**: Read-only SQL query execution with safety validation
- **Connection Pooling**: Efficient database connection management
- **Query Optimization**: Automatic query limits and performance monitoring

### Intelligent Features
- **Context Memory**: Leverages conversation history to avoid redundant database calls
- **Duplicate Call Prevention**: Smart caching system prevents unnecessary repeated operations
- **Multi-Project Support**: Handle multiple Firebase projects simultaneously
- **Data Visualization**: Built-in chart generation capabilities
- **Real-time Analytics**: Live data analysis and pattern recognition

### Smart Chatbot Client
- **Conversation Memory**: Maintains context across interactions
- **Tool Chaining**: Intelligently chains multiple operations based on previous results
- **Schema Integration**: Automatically incorporates database schema into analysis
- **Error Recovery**: Robust error handling and recovery mechanisms

## 🛠️ Technology Stack

- **Backend**: Python with FastMCP framework
- **Firebase**: Firebase Admin SDK for Firestore operations
- **Database**: PostgreSQL with psycopg2 and connection pooling
- **AI Integration**: OpenAI GPT models for intelligent query processing
- **Visualization**: Matplotlib, Seaborn, and Pandas for data visualization
- **Environment**: Docker-ready with environment variable configuration

## 📋 Prerequisites

- Python 3.8 or higher
- Firebase project with service account credentials
- Supabase project with connection string
- OpenAI API key
- Git

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mcp-firebase-supabase-system.git
cd mcp-firebase-supabase-system
```

### 2. Create Virtual Environment

```bash
python -m venv mcp_env
source mcp_env/bin/activate  # On Windows: mcp_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the root directory:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Firebase Configuration
FIREBASE_CRED_JSON=path/to/your/firebase-service-account.json
FIREBASE_PROJECT1_JSON=path/to/another/project/credentials.json  # Optional

# Supabase Configuration
SUPABASE_CONNECTION_STRING=postgresql://user:password@host:port/database

# Optional: Additional Configuration
CHARTS_OUTPUT_PATH=./charts/
LOG_LEVEL=INFO
```

### 5. Firebase Setup

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Create or select your project
3. Navigate to Project Settings → Service Accounts
4. Generate a new private key (JSON file)
5. Save the JSON file and update the path in your `.env` file

### 6. Supabase Setup

1. Go to [Supabase Dashboard](https://supabase.com/dashboard)
2. Create or select your project
3. Go to Settings → Database
4. Copy the connection string
5. Update the `SUPABASE_CONNECTION_STRING` in your `.env` file

### 7. Server Configuration

Create a `server_config.json` file:

```json
{
  "mcpServers": {
    "unified-firebase-supabase": {
      "url": "http://localhost:8005"
    }
  }
}
```

## 🏃‍♂️ Running the System

### 1. Start the MCP Server

```bash
python supa_fire_server.py
```

The server will start on `http://localhost:8005` and display:
```
Starting Database Schema and Read-Only Query MCP Server...
Database URL configured: Yes
```

### 2. Start the Chatbot Client

In a new terminal window:

```bash
python mcp_chatbot.py
```

The chatbot will initialize and display:
```
✅ Schema loaded: X tables, Y columns
MCP Chatbot Started!
Type your query or 'quit' to exit.
```

## 💬 Using the System

### Basic Commands

- **Query databases**: Ask natural language questions about your data
- **`/reset`**: Clear conversation memory
- **`/schema`**: Display current database schema
- **`/memory`**: View conversation context
- **`/cc`**: Clear tool call cache
- **`quit`**: Exit the chatbot

### Example Queries

```
# Firebase Analytics
"Show me statistics for the 'files' collection"
"Analyze file patterns in the documents collection"
"Compare activity between January and February 2024"

# Supabase Queries  
"Show me all users created in the last 30 days"
"What are the most popular product categories?"
"Generate a chart of monthly sales data"

# Cross-platform Analysis
"Compare Firebase file uploads with Supabase user registrations"
"Create a visualization of data trends across both platforms"
```

## 🔧 Available Tools

### Firebase Tools
- `firebase_list_collections` - List all collections in a project
- `firebase_get_collection_stats` - Comprehensive collection analysis
- `firebase_list_files_sample` - Paginated file listing with filters
- `firebase_analyze_file_patterns` - Pattern and trend analysis
- `firebase_get_file_details` - Detailed file information
- `firebase_search_and_filter` - Advanced search capabilities
- `firebase_compare_time_periods` - Time-based comparisons

### Supabase Tools
- `supabase_execute_query` - Safe SQL query execution

### Utility Tools
- `create_data_visualization` - Generate charts and visualizations
- `clear_cache` - Clear tool call cache

## 🛡️ Security Features

- **Read-only Operations**: All database operations are read-only for safety
- **Query Validation**: SQL injection protection with query sanitization
- **Connection Pooling**: Secure and efficient database connections
- **Environment Variables**: Sensitive credentials stored securely
- **Duplicate Call Prevention**: Prevents accidental data overwrites

## 📊 Advanced Features

### Context Memory System
The chatbot intelligently remembers:
- Previous database queries and results
- Collection structures and field types
- Analysis patterns and insights
- User preferences and common operations

### Smart Tool Chaining
- Automatically uses previous results for follow-up queries
- Avoids redundant database calls
- Builds comprehensive analysis from multiple data sources

### Data Visualization
- Automatic chart generation from query results
- Multiple chart types (bar, line, scatter, pie, histogram)
- Customizable styling and output formats

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Verify environment variables are correctly set
   - Check Firebase credentials file paths
   - Ensure Supabase connection string is valid

2. **Permission Issues**
   - Verify Firebase service account has necessary permissions
   - Check Supabase user permissions for database access

3. **Tool Call Errors**
   - Use `/cc` command to clear cache
   - Restart the MCP server if persistent issues occur

## 🔄 Updates & Roadmap

### Upcoming Features
- [ ] Web dashboard interface

### Recent Updates
- ✅ Context memory system
- ✅ Duplicate call prevention
- ✅ Multi-project Firebase support
- ✅ Advanced data visualization
- ✅ Connection pooling optimization

---

**Made by Muhamad Dwirizqy Wimbassa**