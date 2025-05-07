import json
import time
import redis
import re # Import re module
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Union, Callable, Any

# --- Event Dataclass Definitions (Revised Order) ---

@dataclass
class BaseEvent:
    # Base class can be empty or hold fields common to ALL events
    # If adding fields here, ensure they have defaults if any subclass
    # introduces non-default fields.
    pass

@dataclass
class UserPromptEvent(BaseEvent):
    content: str # Non-default first
    event_type: str = "USER_PROMPT"
    timestamp: float = field(default_factory=time.time) # Default field last

@dataclass
class AssistantMessageEvent(BaseEvent):
    content: str # Non-default first
    event_type: str = "ASSISTANT_MESSAGE"
    timestamp: float = field(default_factory=time.time) # Default field last

@dataclass
class AddFileEvent(BaseEvent):
    filepath: str # Non-default first
    read_only: bool = False
    event_type: str = "ADD_FILE"
    timestamp: float = field(default_factory=time.time) # Default field last

@dataclass
class DropFileEvent(BaseEvent):
    filepath: str # Non-default first
    event_type: str = "DROP_FILE"
    timestamp: float = field(default_factory=time.time) # Default field last

@dataclass
class LLMResponseEditEvent(BaseEvent):
    edit_content: str # Non-default first
    target_filepath: Optional[str] = None
    event_type: str = "LLM_RESPONSE_EDIT"
    timestamp: float = field(default_factory=time.time) # Default field last

@dataclass
class ApplyEditEvent(BaseEvent):
    filepaths: List[str] # Non-default first
    commit_hash: Optional[str] = None
    event_type: str = "APPLY_EDIT"
    timestamp: float = field(default_factory=time.time) # Default field last

@dataclass
class UserRejectEditEvent(BaseEvent):
    # Only default fields
    event_type: str = "USER_REJECT_EDIT"
    timestamp: float = field(default_factory=time.time) # Default field last

@dataclass
class RunCommandEvent(BaseEvent):
    command: str # Non-default first
    event_type: str = "RUN_COMMAND"
    timestamp: float = field(default_factory=time.time) # Default field last

@dataclass
class CommandOutputEvent(BaseEvent):
    command: str # Non-default first
    output: str # Non-default first
    exit_status: int = 0
    event_type: str = "COMMAND_OUTPUT"
    timestamp: float = field(default_factory=time.time) # Default field last

@dataclass
class UserCommitEvent(BaseEvent):
    # Only default fields
    message: Optional[str] = None
    event_type: str = "USER_COMMIT"
    timestamp: float = field(default_factory=time.time) # Default field last

@dataclass
class ModeChangeEvent(BaseEvent):
    mode: str # Non-default first
    event_type: str = "MODE_CHANGE"
    timestamp: float = field(default_factory=time.time) # Default field last

@dataclass
class SettingChangeEvent(BaseEvent):
    setting: str # Non-default first
    value: str # Non-default first
    event_type: str = "SETTING_CHANGE"
    timestamp: float = field(default_factory=time.time) # Default field last

@dataclass
class AddWebcontentEvent(BaseEvent):
    url: str # Non-default first
    content: str # Non-default first
    event_type: str = "ADD_WEBCONTENT"
    timestamp: float = field(default_factory=time.time) # Default field last

@dataclass
class PasteContentEvent(BaseEvent):
    type: str # Non-default first
    name: Optional[str] = None
    content: Optional[str] = None
    event_type: str = "PASTE_CONTENT"
    timestamp: float = field(default_factory=time.time) # Default field last

@dataclass
class ClearHistoryEvent(BaseEvent):
    # Only default fields
    event_type: str = "CLEAR_HISTORY"
    timestamp: float = field(default_factory=time.time) # Default field last

@dataclass
class ResetChatEvent(BaseEvent):
    # Only default fields
    event_type: str = "RESET_CHAT"
    timestamp: float = field(default_factory=time.time) # Default field last

# Union type for type hinting if needed later
HistoryEvent = Union[
    UserPromptEvent, AssistantMessageEvent, AddFileEvent, DropFileEvent,
    LLMResponseEditEvent, ApplyEditEvent, UserRejectEditEvent, RunCommandEvent,
    CommandOutputEvent, UserCommitEvent, ModeChangeEvent, SettingChangeEvent,
    AddWebcontentEvent, PasteContentEvent, ClearHistoryEvent, ResetChatEvent
]


# --- Redis History Manager (No changes needed here) ---
# ... (Rest of the RedisHistoryManager class remains the same) ...

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
        verbose: bool = False,
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
            self.verbose = verbose # Store verbose flag
            if self.verbose:
                print(f"RedisHistoryManager initialized. Verbose: {self.verbose}")
            print(f"Connected to Redis. History keys: {self.json_list_key}, {self.text_list_key}")
        except redis.exceptions.ConnectionError as e:
            print(f"Error connecting to Redis: {e}")
            print("History management will not function.")
            self.redis_client = None
            self.verbose = False # Ensure verbose is false if connection fails
        except Exception as e:
             print(f"Unexpected error during Redis connection: {e}")
             self.redis_client = None
             self.verbose = False # Ensure verbose is false on other exceptions


    def _format_event_to_text(self, event: BaseEvent) -> str:
        """Formats an event dataclass into the '[EVENT: TYPE] key=value...' string."""
        event_dict = asdict(event)
        # Get type from the instance itself now
        event_type = event_dict.pop('event_type', 'UNKNOWN') # Pop event_type first
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
        # Default other events (like ADD_FILE, MODE_CHANGE, RUN_COMMAND) to 'user'
        # as they often represent user actions or context provided by the user's side.
        # Alternatively, they could be mapped to 'system' or filtered out later.
        else:
            return "user" # Default assumption

    def generate_llm_context(
        self,
        max_tokens: int,
        tokenizer_func: Callable[[Union[str, List[Dict[str, str]]]], int],
        compaction_level: int = 0
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
        num_total_events = len(all_event_texts)
        for i, event_text_original in enumerate(reversed(all_event_texts)):
            event_text_to_process = event_text_original
            is_event_skipped_by_compaction = False
            original_event_age_rank = i # 0 is most recent, num_total_events-1 is oldest in this reversed iteration

            # --- Apply Compaction Logic Here ---
            if compaction_level >= 1:
                match = re.match(r"\[EVENT: ([\w_]+)\](.*)", event_text_original)
                if match:
                    event_type_str = match.group(1)
                    event_data_str = match.group(2).strip()

                    # Rule 1: Skip trivial assistant messages if they are not "very recent"
                    if event_type_str == "ASSISTANT_MESSAGE":
                        # "Very recent" means original_event_age_rank is small.
                        # Skip if older than the 3rd most recent original event.
                        if original_event_age_rank > 3:
                            content_match = re.search(r"content=(.*)", event_data_str)
                            if content_match:
                                content_val = content_match.group(1).strip()
                                trivial_contents = ["Ok.", "Okay.", "Understood.", "Got it.", "Done."]
                                if content_val in trivial_contents:
                                    is_event_skipped_by_compaction = True
                                    if self.verbose:
                                        print(f"Compaction (L1): Skipped trivial assistant message (age rank {original_event_age_rank}): {event_text_original[:100]}...")
                    
                    # Rule 2: For older CommandOutputEvent, if output is very long, replace with summary
                    elif event_type_str == "COMMAND_OUTPUT":
                        # Skip if older than the 7th most recent original event AND long
                        if original_event_age_rank > 7 and len(event_text_original) > 300:
                            command_match = re.search(r"command=((?:\\.|[^,])+)", event_data_str)
                            status_match = re.search(r"exit_status=(\d+)", event_data_str)
                            
                            cmd_name = command_match.group(1) if command_match else "unknown_command"
                            # Unescape command name if needed, for simple display it's fine
                            cmd_name = cmd_name.replace("\\,", ",").replace("\\\\[", "[").replace("\\\\]", "]")

                            status = status_match.group(1) if status_match else "unknown_status"
                            
                            event_text_to_process = f"[EVENT: COMMAND_OUTPUT] command={cmd_name}, output=[truncated due to age/size], exit_status={status}"
                            if self.verbose:
                                print(f"Compaction (L1): Truncated old command output (age rank {original_event_age_rank}): {event_text_original[:100]}... -> {event_text_to_process}")
            
            if is_event_skipped_by_compaction:
                continue

            role = self._get_role_from_event_text(event_text_to_process)
            message = {"role": role, "content": event_text_to_process}

            # Estimate token count for this message *plus* existing selected
            potential_messages = [message] + selected_messages
            try:
                # Use the tokenizer on the list of messages for better accuracy
                potential_tokens = tokenizer_func(potential_messages)

            except Exception as e:
                 print(f"Warning: Tokenizer function failed during context generation. Error: {e}. Estimating token count.")
                 # Simple estimation if tokenizer fails
                 potential_tokens = current_tokens + len(event_text_to_process.split()) * 2 # Rough estimate

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

# --- Example Usage (Restored) ---
if __name__ == '__main__':
    # This is conceptual and won't run without a Redis server
    # and proper integration into Aider.

    print("Conceptual Example:")
    try:
        # Assume Redis is running on localhost:6379
        history_manager = RedisHistoryManager(session_id='test_session_123')

        if history_manager.redis_client:
            # Clear any previous test data
            history_manager.clear_history()

            # Add some events
            history_manager.add_event(UserPromptEvent(content="Add a function to calculate factorial."))
            history_manager.add_event(AddFileEvent(filepath="math_utils.py"))
            history_manager.add_event(LLMResponseEditEvent(edit_content="```diff\n...fake diff...\n```", target_filepath="math_utils.py"))
            history_manager.add_event(ApplyEditEvent(filepaths=["math_utils.py"], commit_hash="abcdef1"))
            history_manager.add_event(UserPromptEvent(content="Now add tests for it."))
            history_manager.add_event(AddFileEvent(filepath="test_math_utils.py"))
            history_manager.add_event(AssistantMessageEvent(content="Okay, I will add tests."))

            # Define a dummy tokenizer for the example
            def dummy_tokenizer(messages_or_str: Union[str, List[Dict[str, str]]]) -> int:
                if isinstance(messages_or_str, str):
                    return len(messages_or_str.split()) # Very rough word count
                else:
                    count = 0
                    for msg in messages_or_str:
                         content = msg.get('content', '')
                         if isinstance(content, list): # Handle potential multipart content
                             content = " ".join(p.get('text', '') for p in content if p.get('type') == 'text')
                         count += len(content.split())
                    return count

            # Generate context with a token limit
            print("\nGenerating context (limit 50 tokens):")
            context = history_manager.generate_llm_context(max_tokens=50, tokenizer_func=dummy_tokenizer)
            print(json.dumps(context, indent=2))

            print("\nGenerating context (limit 20 tokens):")
            context_truncated = history_manager.generate_llm_context(max_tokens=20, tokenizer_func=dummy_tokenizer)
            print(json.dumps(context_truncated, indent=2))

            # Example: Clear history
            # history_manager.clear_history()
            # print("\nHistory cleared.")
            # context_after_clear = history_manager.generate_llm_context(max_tokens=100, tokenizer_func=dummy_tokenizer)
            # print(json.dumps(context_after_clear, indent=2))

        else:
            print("Cannot run example without Redis connection.")

    except Exception as e:
        print(f"An error occurred during the example: {e}")
