import streamlit as st
import asyncio
import nest_asyncio
import json
from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession
from mcp.client.sse import sse_client
from contextlib import AsyncExitStack

nest_asyncio.apply()
load_dotenv()

class MCP_ChatBot:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI()
        self.available_tools = []
        self.sessions = {}

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
            return True
        except Exception as e:
            st.error(f"❌ Failed to connect to {server_name}: {e}")
            return False

    async def connect_to_servers(self):
        success = False
        with open("server_config.json", "r") as file:
            config = json.load(file)
        for name, server_config in config["mcpServers"].items():
            if await self.connect_to_server(name, server_config):
                success = True
        if not success:
            st.stop()  # Stop Streamlit if no servers are connected

    async def process_query(self, query):
        messages = [{"role": "user", "content": query}]
        response_area = st.chat_message("assistant")

        while True:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.available_tools,
                tool_choice="auto"
            )
            msg = response.choices[0].message

            if msg.content:
                response_area.markdown(msg.content)

            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_msgs = []
                messages.append(msg)

                for tool_call in msg.tool_calls:
                    fname = tool_call.function.name
                    fargs = json.loads(tool_call.function.arguments)
                    tool_id = tool_call.id

                    session = self.sessions.get(fname)
                    if not session:
                        st.warning(f"⚠️ Tool '{fname}' session not found.")
                        continue

                    with st.spinner(f"Calling tool: `{fname}`..."):
                        result = await session.call_tool(fname, arguments=fargs)

                    tool_response = str(result.content)
                    st.chat_message("tool").markdown(f"**Tool `{fname}` result:**\n```\n{tool_response}\n```")

                    tool_msgs.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": fname,
                        "content": tool_response
                    })

                messages.extend(tool_msgs)

                # Final response after tool usage
                followup = self.openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=self.available_tools,
                    tool_choice="auto"
                )
                final_msg = followup.choices[0].message
                if final_msg.content:
                    st.chat_message("assistant").markdown(final_msg.content)
                break
            else:
                break

async def run_chatbot():
    st.set_page_config(page_title="MCP Chatbot", page_icon="🤖")
    st.title("🤖 MCP Chatbot")

    bot = MCP_ChatBot()
    await bot.connect_to_servers()

    if prompt := st.chat_input("Ask something..."):
        st.chat_message("user").write(prompt)
        await bot.process_query(prompt)

if __name__ == "__main__":
    asyncio.run(run_chatbot())