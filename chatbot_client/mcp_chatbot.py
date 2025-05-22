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
