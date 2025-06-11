from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession
from mcp.client.sse import sse_client
from contextlib import AsyncExitStack
import json
import asyncio
import nest_asyncio
import os
import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool
import threading
from contextlib import contextmanager

nest_asyncio.apply()
load_dotenv()

class MCP_ChatBot:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI()
        self.available_tools = []
        self.available_prompts = []
        self.sessions = {}
        self.messages = []
        self.database_schema = None
        self._connection_pool = None
        self._pool_lock = threading.Lock()

    def get_connection_pool(self):
        """Get or create the database connection pool"""
        database_url = os.getenv("SUPABASE_CONNECTION_STRING")
        
        if not database_url:
            return None
        
        with self._pool_lock:
            if self._connection_pool is None:
                try:
                    self._connection_pool = ThreadedConnectionPool(
                        minconn=1,
                        maxconn=3,
                        dsn=database_url
                    )
                except Exception as e:
                    print(f"Failed to create database connection pool: {e}")
                    return None
        
        return self._connection_pool

    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections with connection pooling"""
        pool = self.get_connection_pool()
        if not pool:
            raise Exception("Database connection pool not available")
        
        conn = None
        try:
            conn = pool.getconn()
            yield conn
        finally:
            if conn:
                pool.putconn(conn)

    def execute_safe_query(self, query: str, params: tuple = None):
        """Execute a read-only query safely"""
        with self.get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]

    async def load_database_schema(self, schemas=['public']):
        """Load database schema information"""
        try:
            print("Loading database schema...")
            
            schema_info = {
                "schemas": {},
                "summary": {
                    "total_schemas": 0,
                    "total_tables": 0,
                    "total_columns": 0,
                    "total_foreign_keys": 0
                }
            }
            
            for schema_name in schemas:
                schema_data = await self.get_single_schema_details(schema_name)
                if schema_data:
                    schema_info["schemas"][schema_name] = schema_data
                    
                    # Update summary
                    schema_info["summary"]["total_schemas"] += 1
                    schema_info["summary"]["total_tables"] += len(schema_data.get("tables", {}))
                    schema_info["summary"]["total_columns"] += sum(
                        len(table.get("columns", [])) for table in schema_data.get("tables", {}).values()
                    )
                    schema_info["summary"]["total_foreign_keys"] += len(schema_data.get("foreign_keys", []))
            
            self.database_schema = schema_info
            print(f"✅ Schema loaded: {schema_info['summary']['total_tables']} tables, {schema_info['summary']['total_columns']} columns")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load database schema: {str(e)}")
            self.database_schema = None
            return False

    async def get_single_schema_details(self, schema_name: str):
        """Get detailed information for a single schema"""
        
        # Query for tables and columns
        tables_query = """
            SELECT
                t.table_name,
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default,
                c.ordinal_position,
                CASE
                    WHEN pk.constraint_type = 'PRIMARY KEY' THEN true
                    ELSE false
                END AS is_primary_key
            FROM
                information_schema.tables t
            JOIN
                information_schema.columns c ON t.table_schema = c.table_schema AND t.table_name = c.table_name
            LEFT JOIN (
                SELECT
                    tc.table_schema,
                    tc.table_name,
                    ccu.column_name,
                    tc.constraint_type
                FROM
                    information_schema.table_constraints tc
                JOIN
                    information_schema.constraint_column_usage ccu ON tc.constraint_name = ccu.constraint_name
                WHERE
                    tc.constraint_type = 'PRIMARY KEY'
            ) pk ON t.table_schema = pk.table_schema AND t.table_name = pk.table_name AND c.column_name = pk.column_name
            WHERE
                t.table_schema = %s AND t.table_type = 'BASE TABLE'
            ORDER BY
                t.table_name, c.ordinal_position;
        """
        
        # Query for foreign keys
        fk_query = """
            SELECT
                tc.table_name AS source_table,
                kcu.column_name AS source_column,
                ccu.table_name AS target_table,
                ccu.column_name AS target_column,
                tc.constraint_name
            FROM
                information_schema.table_constraints AS tc
            JOIN
                information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
            JOIN
                information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema
            WHERE
                tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = %s;
        """
        
        try:
            # Get tables and columns
            tables_data = self.execute_safe_query(tables_query, (schema_name,))
            
            # Organize table information
            tables = {}
            for row in tables_data:
                table_name = row['table_name']
                if table_name not in tables:
                    tables[table_name] = {
                        "columns": [],
                        "primary_keys": []
                    }
                
                column_info = {
                    "name": row['column_name'],
                    "data_type": row['data_type'],
                    "is_nullable": row['is_nullable'] == 'YES',
                    "default": row['column_default'],
                    "position": row['ordinal_position'],
                    "is_primary_key": row['is_primary_key']
                }
                
                tables[table_name]["columns"].append(column_info)
                
                if row['is_primary_key']:
                    tables[table_name]["primary_keys"].append(row['column_name'])
            
            # Get foreign keys
            fk_data = self.execute_safe_query(fk_query, (schema_name,))
            foreign_keys = []
            for row in fk_data:
                foreign_keys.append({
                    "constraint_name": row['constraint_name'],
                    "source_table": row['source_table'],
                    "source_column": row['source_column'],
                    "target_table": row['target_table'],
                    "target_column": row['target_column']
                })
            
            return {
                "tables": tables,
                "foreign_keys": foreign_keys
            }
            
        except Exception as e:
            print(f"Error getting schema details for {schema_name}: {e}")
            return {}

    def format_schema_for_prompt(self):
        """Format database schema for system prompt"""
        if not self.database_schema or not self.database_schema.get("schemas"):
            return "No database schema available. Cannot perform Supabase database operations."
        
        content = "# DATABASE SCHEMA INFORMATION\n\n"
        content += "You have access to the following Supabase PostgreSQL database schema. "
        content += "Use this information to construct accurate SQL queries when using database tools.\n\n"
        
        for schema_name, schema_data in self.database_schema["schemas"].items():
            content += f"## Schema: {schema_name}\n\n"
            
            # Tables section
            content += "### Tables and Columns:\n\n"
            for table_name, table_info in schema_data.get("tables", {}).items():
                content += f"**{table_name}**\n"
                
                # Columns
                for col in table_info.get("columns", []):
                    pk_marker = " (PRIMARY KEY)" if col["is_primary_key"] else ""
                    nullable = "NULL" if col["is_nullable"] else "NOT NULL"
                    default = f" DEFAULT {col['default']}" if col["default"] else ""
                    content += f"  - {col['name']}: {col['data_type']}{pk_marker} {nullable}{default}\n"
                
                content += "\n"
            
            # Foreign keys section
            if schema_data.get("foreign_keys"):
                content += "### Foreign Key Relationships:\n"
                for fk in schema_data["foreign_keys"]:
                    content += f"- {fk['source_table']}.{fk['source_column']} → {fk['target_table']}.{fk['target_column']}\n"
                content += "\n"
        
        # Summary
        summary = self.database_schema["summary"]
        content += f"**Summary:** {summary['total_tables']} tables, {summary['total_columns']} columns, {summary['total_foreign_keys']} foreign keys\n\n"
        content += "IMPORTANT: Always reference this schema when constructing SQL queries. If no schema is provided above, inform the user that database operations are not available.\n"
        
        return content

    async def connect_to_server(self, server_name, server_config):
        try:
            sse_transport = await self.exit_stack.enter_async_context(
                sse_client(url=f"{server_config['url']}/sse")
            )
            read, write = sse_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()

            tools_response = await session.list_tools()
            for tool in tools_response.tools:
                self.sessions[tool.name] = session
                self.available_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })

            prompts_response = await session.list_prompts()
            if prompts_response and prompts_response.prompts:
                for prompt in prompts_response.prompts:
                    self.sessions[prompt.name] = session
                    self.available_prompts.append({
                        "name": prompt.name,
                        "description": prompt.description,
                        "arguments": prompt.arguments
                    })

            resources_response = await session.list_resources()
            if resources_response and resources_response.resources:
                for resource in resources_response.resources:
                    self.sessions[str(resource.uri)] = session

        except Exception as e:
            print(f"Error connecting to {server_name}: {e}")

    async def connect_to_servers(self):
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)
            servers = data.get("mcpServers", {})
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server config: {e}")
            raise

    async def call_tool(self, tool_name: str, arguments: dict):
        """Call a specific MCP tool by name"""
        session = self.sessions.get(tool_name)
        if not session:
            return {"success": False, "error": f"Tool '{tool_name}' not found"}
        
        try:
            result = await session.call_tool(tool_name, arguments=arguments)
            # Try to parse the result as JSON for better handling
            try:
                parsed_result = json.loads(str(result.content))
                return parsed_result
            except:
                # If it's not JSON, return as is
                return {"success": True, "result": str(result.content)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_system_prompt(self):
        """Get enhanced system prompt with database schema and context memory emphasis"""
        schema_info = self.format_schema_for_prompt()
        
        system_prompt = f"""You are an intelligent data analysis assistant with access to Firebase and Supabase databases through MCP tools.

## CONTEXT MEMORY PRIORITY
**CRITICAL:** Always leverage conversation history and previous tool results. Build upon prior information instead of starting fresh each time.

## FIREBASE OPERATIONS (No SQL - Use Context!)
For Firebase tools, ALWAYS remember and reference:
- Collection names from previous firebase_list_collections calls
- Field structures from firebase_get_collection_stats results  
- Document IDs and data from firebase_list_files_sample calls
- Analysis patterns from firebase_analyze_file_patterns results

**Firebase Workflow Strategy:**
1. If collection info isn't in memory → Use firebase_list_collections
2. If field structure unknown → Use firebase_get_collection_stats  
3. Build filters/searches using remembered field names and structures
4. Reference previous analysis results to avoid redundant calls

## SUPABASE OPERATIONS (SQL-Based)
{schema_info}

For Supabase queries:
- Use schema information above for accurate SQL construction
- Remember table relationships and join patterns from previous queries
- Build upon previous query results for follow-up analysis

## INTELLIGENT TOOL USAGE
- **Avoid redundant calls** - Check conversation history first
- **Chain operations** - Use results from one tool to inform the next
- **Provide context** - Explain what you remember from previous interactions
- **Be efficient** - Combine multiple insights from single tool calls

## RESPONSE PATTERN
1. Acknowledge what you remember from context
2. Identify what additional information is needed
3. Use tools strategically based on conversation history
4. Synthesize new results with previous findings

Always prioritize using existing context over making new tool calls when possible."""

        return system_prompt

    async def process_query(self, query):
        # Add system prompt as the first message if not already present
        if not self.messages or self.messages[0].get('role') != 'system':
            system_prompt = self.get_system_prompt()
            self.messages.insert(0, {'role': 'system', 'content': system_prompt})
        
        self.messages.append({'role': 'user', 'content': query})

        while True:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.messages,
                max_tokens=2048,
                tools=self.available_tools if self.available_tools else None,
                tool_choice="auto"
            )

            response_message = response.choices[0].message
            has_tool_use = False

            if response_message.content:
                print("\nAI: " + response_message.content)

            if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
                has_tool_use = True
                tool_messages = []

                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    tool_call_id = tool_call.id

                    print(f"\nUsing tool: {function_name}")
                    print(f"Parameters: {json.dumps(function_args, indent=2)}")

                    session = self.sessions.get(function_name)
                    if not session:
                        print(f"Tool '{function_name}' not found.")
                        continue

                    result = await session.call_tool(function_name, arguments=function_args)
                    tool_output = str(result.content)

                    try:
                        tool_output = json.dumps(json.loads(tool_output), indent=2)
                    except:
                        pass

                    print(f"Tool result:\n{tool_output}")

                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": function_name,
                        "content": tool_output
                    })

                self.messages.append(response_message)
                self.messages.extend(tool_messages)
            else:
                self.messages.append(response_message)

            if not has_tool_use:
                break

    async def chat_loop(self):
        print("\nMCP Chatbot Started!")
        print("Type your query or 'quit' to exit.")
        print("Type /reset to clear conversation memory.")
        print("Type /cc to clear tool call cache.")
        print("Type /schema to display current database schema.")
        print("Type /memory to see conversation context.")
        
        while True:
            try:
                query = input("\n--------------------------------\nQuery: ").strip()
                if not query:
                    continue
                if query.lower() == "quit":
                    break
                if query.lower() == "/reset":
                    self.messages = []
                    print("✅ Memory has been reset.")
                    continue
                if query.lower() == "/schema":
                    if self.database_schema:
                        print("\n" + self.format_schema_for_prompt())
                    else:
                        print("❌ No database schema loaded.")
                    continue
                if query.lower() == "/memory":
                    print(f"\n📝 Conversation has {len(self.messages)} messages")
                    print("Recent context:")
                    for msg in self.messages[-5:]:
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')[:100] + "..." if len(msg.get('content', '')) > 100 else msg.get('content', '')
                        print(f"  {role}: {content}")
                    continue
                if query.lower() == "/cc":
                    # Try to use the MCP tool first
                    if "clear_cache" in self.sessions:
                        try:
                            result = await self.call_tool("clear_cache", {})
                            print(f"✅ Server cache cleared: {result}")
                        except Exception as e:
                            print(f"❌ Error clearing server cache: {e}")
                    else:
                        print("❌ clear_cache tool not available on server.")
                    continue
                await self.process_query(query)
            except Exception as e:
                print(f"Error: {e}")

    async def cleanup(self):
        await self.exit_stack.aclose()
        # Cleanup database connections
        if self._connection_pool:
            with self._pool_lock:
                self._connection_pool.closeall()

async def main():
    chatbot = MCP_ChatBot()
    try:
        # Load database schema first
        await chatbot.load_database_schema()
        
        # Then connect to MCP servers
        await chatbot.connect_to_servers()
        
        # Start chat loop
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()

if __name__ == "__main__":
    asyncio.run(main())