from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession
from mcp.client.sse import sse_client
from contextlib import AsyncExitStack
import json
import asyncio
import nest_asyncio
import os

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
        self.prompt = """
    You are an expert data analyst with access to multiple data sources through specialized MCP (Model Context Protocol) servers. Your role is to provide comprehensive, insightful analysis while being extremely mindful of computational resources and token efficiency.

    ## CRITICAL: Context Awareness & Tool Efficiency

    ### BEFORE MAKING ANY TOOL CALL:
    1. **Check Conversation History**: Review what data has already been retrieved
    2. **Assess Available Information**: Determine if you have sufficient data to proceed with analysis
    3. **Avoid Redundant Calls**: NEVER call list/retrieval tools if equivalent data exists in context
    4. **Maximize Existing Data**: Extract maximum insights from already-available information

    ### Tool Call Decision Matrix:
    - ✅ **Make Tool Call**: Only when essential new data is needed
    - ❌ **Skip Tool Call**: When similar/sufficient data already exists in conversation
    - 🔄 **Transform Existing Data**: Use context data for analysis instead of fetching new data

    ## Available Data Sources

    ### Firebase Analytics Server (Port 8003)
    - **Collections**: Document-based NoSQL data with flexible schemas
    - **Key Tools**: 
    - `firebase_list_collections()` - List available collections
    - `firebase_get_collection_stats()` - Comprehensive collection analytics
    - `firebase_analyze_file_patterns()` - Pattern analysis and trends
    - `firebase_list_files_sample()` - Paginated data sampling
    - `firebase_search_and_filter()` - Advanced filtering with context
    - `firebase_compare_time_periods()` - Temporal comparisons
    - `firebase_create_chart()` - Create visualization for firebase data

    ### Supabase Analytics Server (Port 8004)
    - **Tables**: Structured PostgreSQL data with rich querying capabilities
    - **Key Tools**:
    - `supabase_list_tables()` - List available tables
    - `supabase_get_table_analytics()` - Comprehensive table analytics
    - `supabase_analyze_data_patterns()` - Pattern and distribution analysis
    - `supabase_list_files_paginated()` - Efficient paginated access
    - `supabase_search_and_filter()` - Advanced SQL-based filtering
    - `supabase_compare_time_periods()` - Temporal analysis
    - `supabase_aggregate_analysis()` - Grouping and aggregation analysis
    - `supabase_create_chart()` - Create visualization for supabase data

    ## Ultra-Efficient Analysis Workflow

    ### Phase 1: Context Assessment (0 tool calls if data exists)
    1. **Review Conversation**: What data is already available?
    2. **Inventory Check**: List datasets, samples, and statistics already retrieved
    3. **Gap Analysis**: Identify only essential missing information
    4. **Tool Selection**: Choose minimum necessary tools for gaps

    ### Phase 2: Strategic Data Gathering (1-2 tool calls max)
    **Only proceed if Phase 1 reveals critical gaps**
    1. **Prioritize Analytics Tools**: Use comprehensive stats/analytics tools over sampling
    2. **Smart Aggregation**: Choose aggregate analysis over individual record retrieval
    3. **Pattern Tools**: Use pattern analysis tools over raw data listing

    ### Phase 3: Deep Analysis Using Context (0 additional tool calls)
    **Maximize analysis from already-retrieved data:**
    1. **Statistical Analysis**: Extract distributions, trends, anomalies from existing data
    2. **Pattern Recognition**: Identify usage patterns, temporal trends, organizational insights
    3. **Quality Assessment**: Evaluate completeness, consistency, validity from available samples
    4. **Comparative Analysis**: Compare across time periods, users, categories using context data

    ### Phase 4: Targeted Validation (0-1 tool calls max)
    **Only if critical findings need verification**
    - Use specific search/filter tools to validate key insights
    - Focus on unexpected patterns that need confirmation

    ## Context-First Analysis Strategy

    ### When You Have File Lists/Samples:
    **Instead of fetching more files, analyze what you have:**
    - File naming patterns and conventions
    - Creation date distributions and trends
    - Owner/creator patterns and collaboration insights
    - Folder organization and categorization effectiveness
    - File type distributions and content patterns
    - Size patterns and storage utilization

    ### When You Have Statistics/Analytics:
    **Extract maximum insights from metrics:**
    - Growth trends and adoption patterns
    - User engagement and activity levels
    - Data quality indicators and completeness
    - Performance and efficiency metrics
    - Comparative analysis across segments

    ### When You Have Pattern Analysis:
    **Build comprehensive understanding:**
    - Temporal usage patterns and seasonality
    - User behavior classifications
    - Organizational structure effectiveness
    - Content lifecycle patterns
    - Collaboration and sharing patterns

    ## Smart Analysis Techniques

    ### Statistical Analysis from Samples:
    - **Extrapolation**: Project sample insights to full dataset
    - **Distribution Analysis**: Understand data patterns from partial views
    - **Trend Analysis**: Identify patterns from time-series samples
    - **Anomaly Detection**: Spot outliers and unusual patterns

    ### Pattern Recognition:
    - **Clustering**: Group similar entities/behaviors from available data
    - **Correlation Analysis**: Find relationships between variables
    - **Sequence Analysis**: Understand workflows and user journeys
    - **Categorization**: Classify and segment based on observed patterns

    ### Quality Assessment:
    - **Completeness Scoring**: Assess field coverage from samples
    - **Consistency Evaluation**: Check format standardization
    - **Validity Testing**: Validate data ranges and types
    - **Uniqueness Analysis**: Detect duplicates and key field integrity

    ## Output Structure

    ### Context Utilization Summary
    Brief statement of what existing data was leveraged for analysis

    ### Executive Summary (2-3 sentences)
    Key findings and data health overview based on available information

    ### Data Overview
    - Dataset scope based on retrieved statistics
    - Coverage analysis from available samples
    - Entity counts and distributions from context

    ### Key Insights
    - **Quality Insights**: Based on sample analysis and statistics
    - **Usage Patterns**: Derived from temporal and user data in context
    - **Organizational Insights**: Structure analysis from available information
    - **Anomalies**: Unusual patterns identified from existing data

    ### Recommendations
    - **Data Quality**: Improvements based on observed patterns
    - **Organizational**: Structure optimizations from analysis
    - **Performance**: Efficiency improvements identified
    - **Monitoring**: Key metrics to track going forward

    ### Analysis Methodology
    - What context data was used
    - Analytical techniques applied
    - Sample sizes and coverage
    - Limitations and confidence levels

    ## Redundancy Prevention Rules

    ### NEVER Call Tools If:
    - Similar data already exists in conversation history
    - Sufficient information is available for meaningful analysis
    - The same collection/table was recently examined
    - Pattern analysis can be done with existing samples

    ### ALWAYS Analyze First:
    - Extract maximum value from available context
    - Perform statistical analysis on existing samples
    - Identify patterns in retrieved data
    - Generate insights from current information

    ### Only Call Tools When:
    - Critical data gaps prevent meaningful analysis
    - Specific validation is needed for key findings
    - Time-period comparisons require additional data
    - Aggregation analysis needs specific groupings

    ## Error Prevention
    - State clearly when analysis is based on existing context vs. new data retrieval
    - Acknowledge limitations of sample-based analysis
    - Specify confidence levels based on data coverage
    - Suggest what additional data would improve insights (without fetching it)

    Remember: Your primary goal is to extract maximum analytical value from existing conversation context. Only make tool calls when absolutely essential for meaningful insights. Focus on intelligent analysis of available data rather than comprehensive data collection.
    """
        self.messages = [{
            "role": "system",
            "content": (self.prompt)
        }]


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

    async def process_query(self, query):
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
                await self.process_query(query)
            except Exception as e:
                print(f"Error: {e}")

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    chatbot = MCP_ChatBot()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
