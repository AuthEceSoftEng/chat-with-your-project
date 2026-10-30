import uuid

class UniversalBuffer:
    def __init__(self, max_size: int = 50, eviction_policy: str = "lru"):
        self._store = {}
        self._history = []
        self._max_size = max_size
        
        policy = eviction_policy.lower()
        if policy not in ("fifo", "lru"):
            raise ValueError("Eviction policy must be either 'fifo' or 'lru'.")
        self._eviction_policy = policy

    def put(self, data: any) -> str:
        buf_id = f"buf_{str(uuid.uuid4())[:6]}"
        self._store[buf_id] = data
        self._history.append(buf_id)
        
        if len(self._history) > self._max_size:
            oldest = self._history.pop(0)
            self._store.pop(oldest, None)
            
        return buf_id

    def get(self, buf_id: str = "latest") -> any:
        if buf_id == "latest" and self._history:
            buf_id = self._history[-1]
        
        if buf_id not in self._store:
            raise ValueError(f"Buffer reference '{buf_id}' not found or expired.")
        
        if self._eviction_policy == "lru":
            self._history.remove(buf_id)
            self._history.append(buf_id)
            
        return self._store[buf_id]

buffer_registry = UniversalBuffer()