import json
from typing import Any

from task.tools.base import BaseTool
from task.tools.memory.memory_store import LongTermMemoryStore
from task.tools.models import ToolCallParams


class SearchMemoryTool(BaseTool):
    """
    Tool for searching long-term memories about the user.

    Performs semantic search over stored memories to find relevant information.
    """

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store

    @property
    def name(self) -> str:
        return "search_long_term_memory"

    @property
    def description(self) -> str:
        return (
            "Search the user's long-term memories using semantic similarity. "
            "Use BEFORE answering questions that depend on user-specific context such as "
            "location, weather, clothing, schedule, preferences, work, family, or past statements. "
            "Also use when starting a new conversation if the request may relate to known user facts. "
            "Examples: 'What should I wear?' -> search for location/climate preferences; "
            "'What's the weather?' -> search for where the user lives. "
            "Do not guess personal details—retrieve them from memory first."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Can be a question or keywords to find relevant memories",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of most relevant memories to return.",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        query = arguments["query"]
        top_k = arguments.get("top_k", 5)

        results = await self.memory_store.search_memories(
            api_key=tool_call_params.api_key,
            query=query,
            top_k=top_k,
        )

        if not results:
            final_result = "No memories found."
        else:
            lines = ["## Retrieved memories\n"]
            for memory in results:
                lines.append(f"- **{memory.category}**: {memory.content}")
                if memory.topics:
                    lines.append(f"  - Topics: {', '.join(memory.topics)}")
            final_result = "\n".join(lines)

        tool_call_params.stage.append_content(f"{final_result}\n")
        return final_result
