# aider_history_redis.py

import json
import time
import redis
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Union, Callable, Any

# --- Event Dataclass Definitions ---

@dataclass
class BaseEvent:
    timestamp: float = field(default_factory=time.time)
    # Sequence ID could be added later if needed, managed by Redis potentially

@dataclass
class UserPromptEvent(BaseEvent):
    content: str
    event_type: str = "USER_PROMPT"

@dataclass
class AssistantMessageEvent(BaseEvent):
    content: str
    event_type: str = "ASSISTANT_MESSAGE"

@dataclass
class AddFileEvent(BaseEvent):
    filepath: str
    read_only: bool = False # Flag to distinguish /add vs /read-only
    event_type: str = "ADD_FILE" # Combined ADD_FILE and ADD_READONLY_FILE

@dataclass
class DropFileEvent(BaseEvent):
    filepath: str
    event_type: str = "DROP_FILE"

@dataclass
class LLMResponseEditEvent(BaseEvent):
    # Store the raw diff/edit block string as proposed by LLM
    edit_content: str
    # Potentially store target file if easily parsable, else None
    target_filepath: Optional[str] = None
    event_type: str = "LLM_RESPONSE_EDIT"

@dataclass
class ApplyEditEvent(BaseEvent):
    # Filepaths affected by the applied edit
    filepaths: List[str]
    commit_hash: Optional[str] = None
    event_type: str = "APPLY_EDIT"

@dataclass
class UserRejectEditEvent(BaseEvent):
    # No extra data needed, just marks rejection of preceding LLM_RESPONSE_EDIT
    event_type: str = "USER_REJECT_EDIT"

@dataclass
class RunCommandEvent(BaseEvent):
    command: str
    event_type: str = "RUN_COMMAND"

@dataclass
class CommandOutputEvent(BaseEvent):
    command: str
    # Store truncated output
    output: str
    exit_status: int = 0 # Add exit status
    event_type: str = "COMMAND_OUTPUT"

@dataclass
class UserCommitEvent(BaseEvent):
    message: Optional[str] = None
    event_type: str = "USER_COMMIT"

@dataclass
class ModeChangeEvent(BaseEvent):
    mode: str # e.g., "Code", "Ask", "Architect"
    event_type: str = "MODE_CHANGE"

@dataclass
class SettingChangeEvent(BaseEvent):
    setting: str # e.g., "main_model", "reasoning_effort"
    value: str
    event_type: str = "SETTING_CHANGE"

@dataclass
class AddWebcontentEvent(BaseEvent):
    url: str
    content: str # Store truncated content
    event_type: str = "ADD_WEBCONTENT"

@dataclass
class PasteContentEvent(BaseEvent):
    type: str # "text" or "image"
    name: Optional[str] = None
    content: Optional[str] = None # Text content or marker like "[IMAGE: name]"
    event_type: str = "PASTE_CONTENT"

@dataclass
class ClearHistoryEvent(BaseEvent):
    event_type: str = "CLEAR_HISTORY"

@dataclass
class ResetChatEvent(BaseEvent):
    event_type: str = "RESET_CHAT"

# Union type for type hinting if needed later
HistoryEvent = Union[
    UserPromptEvent, AssistantMessageEvent, AddFileEvent, DropFileEvent,
    LLMResponseEditEvent, ApplyEditEvent, UserRejectEditEvent, RunCommandEvent,
    CommandOutputEvent, UserCommitEvent, ModeChangeEvent, SettingChangeEvent,
    AddWebcontentEvent, PasteContentEvent, ClearHistoryEvent, ResetChatEvent
]


# --- Redis History Manager ---

class RedisHistoryManager:
    """
    Manages conversation history using Redis, storing both structured JSON
    and pre-formatted text representations.
    """
    def __init__(
        self,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        redis_db: int = 0,
        session_id: str = 'default_session', # Use a unique ID per chat session
        redis_password: Optional[str] = None,
    ):
        """
        Initializes the Redis connection and sets keys for the session.

        Args:
            redis_host: Hostname of the Redis server.
            redis_port: Port of the Redis server.
            redis_db: Redis database number.
            session_id: A unique identifier for the current chat session.
            redis_password: Password for Redis authentication (optional).
        """
        self.json_list_key = f"aider_history:{session_id}:json"
        self.text_list_key = f"aider_history:{session_id}:text"
        self.session_id = session_id
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=True # Decode responses to strings
            )
            # Test connection
            self.redis_client.ping()
            print(f"Connected to Redis. History keys: {self.json_list_key}, {self.text_list_key}")
        except redis.exceptions.ConnectionError as e:
            print(f"Error connecting to Redis: {e}")
            print("History management will not function.")
            self.redis_client = None
        except Exception as e:
             print(f"Unexpected error during Redis connection: {e}")
             self.redis_client = None


    def _format_event_to_text(self, event: BaseEvent) -> str:
        """Formats an event dataclass into the '[EVENT: TYPE] key=value...' string."""
        event_dict = asdict(event)
        event_type = event_dict.pop('event_type', 'UNKNOWN')
        timestamp = event_dict.pop('timestamp', None) # Exclude timestamp from text format

        parts = []
        for key, value in event_dict.items():
            # Simple representation for lists/None
            if isinstance(value, list):
                value_str = f"[{','.join(map(str, value))}]" if value else "[]"
            elif value is None:
                value_str = "None"
            else:
                # Truncate long strings (e.g., content, output, edit_content)
                value_str = str(value)
                if key in ('content', 'output', 'edit_content') and len(value_str) > 200:
                     value_str = value_str[:100] + "..." + value_str[-100:]

            # Basic escaping for problematic characters if needed (can be expanded)
            value_str = value_str.replace('\n', '\\n').replace(']', '\\]').replace('[', '\\[').replace(',', '\\,')
            parts.append(f"{key}={value_str}")

        data_str = ", ".join(parts)
        return f"[EVENT: {event_type}] {data_str}"

    def add_event(self, event: BaseEvent):
        """
        Adds an event to the history, storing both JSON and formatted text.

        Args:
            event: The event dataclass instance to add.
        """
        if not self.redis_client:
            # Optionally log a warning or raise an error if Redis isn't available
            # print("Warning: Redis client not available. Event not added.")
            return

        try:
            # 1. Serialize to JSON
            event_dict = asdict(event)
            json_string = json.dumps(event_dict)

            # 2. Format to text string
            text_string = self._format_event_to_text(event)

            # 3. Add to Redis lists
            # Use a pipeline for atomicity (optional but good practice)
            pipe = self.redis_client.pipeline()
            pipe.rpush(self.json_list_key, json_string)
            pipe.rpush(self.text_list_key, text_string)
            pipe.execute()

        except redis.exceptions.RedisError as e:
            print(f"Error adding event to Redis: {e}")
        except TypeError as e:
            print(f"Error serializing event: {e}. Event: {event}")
        except Exception as e:
            print(f"Unexpected error adding event: {e}")

    def _get_role_from_event_text(self, event_text: str) -> str:
        """Determines the LLM role based on the event type string."""
        # Simple logic based on event type prefix
        if event_text.startswith("[EVENT: USER_"): # USER_PROMPT, USER_COMMIT, USER_REJECT_EDIT
            return "user"
        elif event_text.startswith("[EVENT: LLM_RESPONSE_EDIT]") or \
             event_text.startswith("[EVENT: ASSISTANT_MESSAGE]"):
            return "assistant"
        # Treat system/agent actions (file adds, commands, mode changes) as user context for the LLM
        else:
            return "user"

    def generate_llm_context(
        self,
        max_tokens: int,
        tokenizer_func: Callable[[Union[str, List[Dict[str, str]]]], int],
        # compaction_level: int = 0 # Placeholder for future compaction logic
    ) -> List[Dict[str, str]]:
        """
        Generates the list of message dictionaries for the LLM context,
        truncating from the start (oldest) if necessary.

        Args:
            max_tokens: The maximum number of tokens allowed for the history part.
            tokenizer_func: A function that takes messages (str or list of dicts)
                            and returns the token count.
            compaction_level: (Future use) Level of compaction to apply.

        Returns:
            A list of message dictionaries [{'role': ..., 'content': ...}].
        """
        if not self.redis_client:
            # print("Warning: Redis client not available. Returning empty context.")
            return []

        try:
            # Retrieve all formatted text events
            # LRANGE is inclusive, 0 to -1 gets the whole list
            all_event_texts = self.redis_client.lrange(self.text_list_key, 0, -1)
        except redis.exceptions.RedisError as e:
            print(f"Error retrieving history from Redis: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error retrieving history from Redis: {e}")
            return []


        selected_messages: List[Dict[str, str]] = []
        current_tokens = 0
        final_event_count = 0

        # Iterate backwards (most recent first)
        for event_text in reversed(all_event_texts):
            # --- Apply Compaction Logic Here (Future) ---
            # Example: Skip simple assistant 'Ok.' messages if compaction level > 0
            # if compaction_level > 0 and event_text == "[EVENT: ASSISTANT_MESSAGE] content=Ok.":
            #     continue

            role = self._get_role_from_event_text(event_text)
            message = {"role": role, "content": event_text}

            # Estimate token count for this message *plus* existing selected
            potential_messages = [message] + selected_messages
            try:
                # Use the tokenizer on the list of messages for better accuracy
                potential_tokens = tokenizer_func(potential_messages)

            except Exception as e:
                 print(f"Warning: Tokenizer function failed during context generation. Error: {e}. Estimating token count.")
                 # Simple estimation if tokenizer fails
                 potential_tokens = current_tokens + len(event_text.split()) * 2 # Rough estimate

            if potential_tokens <= max_tokens:
                selected_messages.insert(0, message) # Insert at beginning to maintain order
                current_tokens = potential_tokens # Update token count based on list
                final_event_count += 1
            else:
                # Stop adding older messages once token limit is reached
                print(f"History truncated for session '{self.session_id}'. Used {final_event_count}/{len(all_event_texts)} events, ~{current_tokens} tokens.")
                break

        return selected_messages

    def clear_history(self):
        """Deletes the history lists for the current session from Redis."""
        if not self.redis_client:
            # print("Warning: Redis client not available. History not cleared.")
            return
        try:
            deleted_count = self.redis_client.delete(self.json_list_key, self.text_list_key)
            print(f"Cleared history for session '{self.session_id}'. Deleted {deleted_count} Redis keys.")
        except redis.exceptions.RedisError as e:
            print(f"Error clearing history in Redis: {e}")
        except Exception as e:
            print(f"Unexpected error clearing history: {e}")


    def reset_chat(self):
        """Resets the chat history (alias for clear_history)."""
        self.clear_history()
