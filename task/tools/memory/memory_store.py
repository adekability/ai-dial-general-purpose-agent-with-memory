import os
os.environ['OMP_NUM_THREADS'] = '1'

import json
from datetime import datetime, UTC, timedelta
import numpy as np
import faiss
from aidial_client import AsyncDial
from sentence_transformers import SentenceTransformer

from task.tools.memory._models import Memory, MemoryData, MemoryCollection


class LongTermMemoryStore:
    """
    Manages long-term memory storage for users.

    Storage format: Single JSON file per user in DIAL bucket
    - File: {user_id}/long-memories.json
    - Caching: In-memory cache with conversation_id as key
    - Deduplication: O(n log n) using FAISS batch search
    """

    DEDUP_INTERVAL_HOURS = 24
    SIMILARITY_THRESHOLD = 0.75

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self._cache: dict[str, MemoryCollection] = {}
        faiss.omp_set_num_threads(1)

    def _create_dial_client(self, api_key: str) -> AsyncDial:
        return AsyncDial(
            base_url=self.endpoint,
            api_key=api_key,
            api_version='2025-01-01-preview',
        )

    async def _get_memory_file_path(self, dial_client: AsyncDial) -> str:
        """Get the path to the memory file in DIAL bucket."""
        appdata_home = await dial_client.my_appdata_home()
        if appdata_home is None:
            raise ValueError("Application appdata home is not available for this user.")
        return f"files/{(appdata_home / '__long-memories' / 'data.json').as_posix()}"

    async def _load_memories(self, api_key: str) -> MemoryCollection:
        dial_client = self._create_dial_client(api_key)
        file_path = await self._get_memory_file_path(dial_client)

        if file_path in self._cache:
            return self._cache[file_path]

        try:
            response = await dial_client.files.download(url=file_path)
            content = await response.aget_content()
            data = json.loads(content.decode('utf-8'))
            memories = MemoryCollection.model_validate(data)
        except Exception:
            memories = MemoryCollection(
                memories=[],
                updated_at=datetime.now(UTC),
                last_deduplicated_at=None,
            )

        self._cache[file_path] = memories
        return memories

    async def _save_memories(self, api_key: str, memories: MemoryCollection):
        """Save memories to DIAL bucket and update cache."""
        dial_client = self._create_dial_client(api_key)
        file_path = await self._get_memory_file_path(dial_client)

        memories.updated_at = datetime.now(UTC)
        json_str = memories.model_dump_json()
        self._cache[file_path] = memories
        await dial_client.files.upload(
            url=file_path,
            file=json_str.encode('utf-8'),
        )

    async def add_memory(self, api_key: str, content: str, importance: float, category: str, topics: list[str]) -> str:
        """Add a new memory to storage."""
        memories = await self._load_memories(api_key)
        embedding = self.model.encode([content])[0].tolist()

        memory = Memory(
            data=MemoryData(
                id=int(datetime.now(UTC).timestamp()),
                content=content,
                importance=importance,
                category=category,
                topics=topics,
            ),
            embedding=embedding,
        )
        memories.memories.append(memory)
        await self._save_memories(api_key, memories)
        return f"Memory stored successfully: {content}"

    async def search_memories(self, api_key: str, query: str, top_k: int = 5) -> list[MemoryData]:
        """
        Search memories using semantic similarity.

        Returns:
            List of MemoryData objects (without embeddings)
        """
        collection = await self._load_memories(api_key)
        if not collection.memories:
            return []

        if self._needs_deduplication(collection):
            collection = await self._deduplicate_and_save(api_key, collection)

        embeddings = np.array([memory.embedding for memory in collection.memories], dtype=np.float32)
        query_embedding = self.model.encode([query]).astype(np.float32)

        faiss.normalize_L2(embeddings)
        faiss.normalize_L2(query_embedding)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        k = min(top_k, len(collection.memories))
        _, indices = index.search(query_embedding, k=k)

        results: list[MemoryData] = []
        for idx in indices[0]:
            if idx < 0:
                continue
            results.append(collection.memories[idx].data)
        return results

    def _needs_deduplication(self, collection: MemoryCollection) -> bool:
        """Check if deduplication is needed (>24 hours since last deduplication)."""
        if len(collection.memories) <= 10:
            return False
        if collection.last_deduplicated_at is None:
            return True
        return datetime.now(UTC) - collection.last_deduplicated_at > timedelta(hours=self.DEDUP_INTERVAL_HOURS)

    async def _deduplicate_and_save(self, api_key: str, collection: MemoryCollection) -> MemoryCollection:
        """
        Deduplicate memories synchronously and save the result.
        Returns the updated collection.
        """
        collection.memories = self._deduplicate_fast(collection.memories)
        collection.last_deduplicated_at = datetime.now(UTC)
        await self._save_memories(api_key, collection)
        return collection

    def _deduplicate_fast(self, memories: list[Memory]) -> list[Memory]:
        """
        Fast deduplication using FAISS batch search with cosine similarity.

        Strategy:
        - Find k nearest neighbors for each memory using cosine similarity
        - Mark duplicates based on similarity threshold (cosine similarity > 0.75)
        - Keep memory with higher importance
        """
        n = len(memories)
        if n <= 1:
            return memories

        embeddings = np.array([memory.embedding for memory in memories], dtype=np.float32)
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        k = min(n, 10)
        similarities, neighbor_indices = index.search(embeddings, k)

        removed: set[int] = set()
        for i in range(n):
            if i in removed:
                continue
            for rank in range(k):
                j = int(neighbor_indices[i][rank])
                if j == i or j in removed:
                    continue
                if similarities[i][rank] <= self.SIMILARITY_THRESHOLD:
                    continue
                if memories[i].data.importance >= memories[j].data.importance:
                    removed.add(j)
                else:
                    removed.add(i)
                    break

        return [memory for idx, memory in enumerate(memories) if idx not in removed]

    async def delete_all_memories(self, api_key: str) -> str:
        """
        Delete all memories for the user.

        Removes the memory file from DIAL bucket and clears the cache
        for the current conversation.
        """
        dial_client = self._create_dial_client(api_key)
        file_path = await self._get_memory_file_path(dial_client)

        self._cache.pop(file_path, None)
        try:
            await dial_client.files.delete(url=file_path)
        except Exception:
            pass

        return "All long-term memories have been deleted successfully."
