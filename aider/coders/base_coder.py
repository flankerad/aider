#!/usr/bin/env python

import base64
import datetime # Added
import hashlib
import json     
import locale
import math
import mimetypes
import os       
import platform
import re
import sys
import threading
import time
import traceback
from collections import defaultdict
from datetime import datetime
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import List, Optional, Dict, Any 

from aider import __version__, models, prompts, urls, utils
from aider.analytics import Analytics
from aider.commands import Commands
from aider.exceptions import LiteLLMExceptions
# from aider.history import ChatSummary # Removed
from aider.io import ConfirmGroup, InputOutput
from aider.linter import Linter
from aider.llm import litellm
from aider.models import RETRY_TIMEOUT
from aider.reasoning_tags import (
    REASONING_TAG,
    format_reasoning_content,
    remove_reasoning_content,
    replace_reasoning_tags,
)
from aider.repo import ANY_GIT_ERROR, GitRepo
from aider.repomap import RepoMap
from aider.run_cmd import run_cmd
from aider.utils import format_content, format_messages, format_tokens, is_image_file
from aider.aider_history_redis import (
    RedisHistoryManager, BaseEvent, UserPromptEvent, AssistantMessageEvent,
    LLMResponseEditEvent, ApplyEditEvent, UserRejectEditEvent, AddFileEvent,
    DropFileEvent, RunCommandEvent, CommandOutputEvent, UserCommitEvent,
    ModeChangeEvent, SettingChangeEvent, AddWebcontentEvent, PasteContentEvent,
    ClearHistoryEvent, ResetChatEvent
    # Note: FileContentEvent, EditOutcomeEvent, LLMFunctionCallEvent, LLMFunctionResponseEvent
    # are NOT in the provided aider_history_redis.py, so they are not imported here.
)


from ..dump import dump  # noqa: F401
from .chat_chunks import ChatChunks

# If llmlingua is used, import it conditionally (Kept for potential future use)
try:
    from llmlingua import PromptCompressor
except ImportError:
    PromptCompressor = None  # Keep None if not installed


class UnknownEditFormat(ValueError):
    def __init__(self, edit_format, valid_formats):
        self.edit_format = edit_format
        self.valid_formats = valid_formats
        super().__init__(
            f"Unknown edit format {edit_format}. Valid formats are: {', '.join(valid_formats)}"
        )


class MissingAPIKeyError(ValueError):
    pass


class FinishReasonLength(Exception):
    pass


def wrap_fence(name):
    return f"<{name}>", f"</{name}>"


all_fences = [
    ("`" * 3, "`" * 3),
    ("`" * 4, "`" * 4),  # LLMs ignore and revert to triple-backtick, causing #2879
    wrap_fence("source"),
    wrap_fence("code"),
    wrap_fence("pre"),
    wrap_fence("codeblock"),
    wrap_fence("sourcecode"),
]


class Coder:
    abs_fnames = None
    abs_read_only_fnames = None
    repo = None
    last_aider_commit_hash = None
    aider_edited_files = None
    last_asked_for_commit_time = 0
    repo_map = None
    functions = None
    num_exhausted_context_windows = 0
    num_malformed_responses = 0
    last_keyboard_interrupt = None
    num_reflections = 0
    max_reflections = 3
    edit_format = None
    yield_stream = False
    temperature = None
    auto_lint = True
    auto_test = False
    test_cmd = None
    lint_outcome = None
    test_outcome = None
    multi_response_content = ""
    partial_response_content = ""
    # commit_before_message = []
    # cur_messages = None 
    # done_messages = None 
    # summarized_done_messages = None 
    # summarizing_messages = None 
    # summarizer_thread = None 
    message_cost = 0.0
    message_tokens_sent = 0
    message_tokens_received = 0
    add_cache_headers = False
    cache_warming_thread = None
    num_cache_warming_pings = 0
    suggest_shell_commands = True
    detect_urls = True
    ignore_mentions = None
    chat_language = None
    file_watcher = None
    history_manager: Optional[RedisHistoryManager] = None

    # llmlingua attributes (kept for potential future use)
    llmlingua_compressor = None
    compression_output_counter = 0
    compression_output_dir = "llmlingua_outputs"


    @classmethod
    def create(
        cls, 
        main_model=None,
        edit_format=None,
        io=None,
        from_coder=None,
        # summarize_from_coder=True, # REMOVED
        history_manager=None, # ENSURED
        **kwargs,
    ):
        import aider.coders as coders

        if not main_model:
            if from_coder:
                main_model = from_coder.main_model
            else:
                main_model = models.Model(models.DEFAULT_MODEL_NAME)

        if edit_format == "code": # Special case for "code" mode
            edit_format = main_model.edit_format # Use model's default
        elif edit_format is None: # If no edit_format specified
            if from_coder:
                edit_format = from_coder.edit_format
            else: # Default to model's edit format if no prior coder and no explicit format
                edit_format = main_model.edit_format


        if not io and from_coder:
            io = from_coder.io

        if not history_manager and from_coder:
             history_manager = from_coder.history_manager

        if from_coder:
            use_kwargs = dict(from_coder.original_kwargs) 

            update = dict(
                fnames=list(from_coder.abs_fnames),
                read_only_fnames=list(from_coder.abs_read_only_fnames),
                # done_messages and cur_messages REMOVED
                aider_commit_hashes=from_coder.aider_commit_hashes,
                commands=from_coder.commands.clone(), # Clone commands object
                total_cost=from_coder.total_cost,
                ignore_mentions=from_coder.ignore_mentions,
                file_watcher=from_coder.file_watcher,
                history_manager=history_manager, 
            )
            use_kwargs.update(update) 
            use_kwargs.update(kwargs) 

            kwargs = use_kwargs
            from_coder.ok_to_warm_cache = False
        
        kwargs['history_manager'] = history_manager # Ensure it's in kwargs

        for coder_class in coders.__all__:
            if hasattr(coder_class, "edit_format") and coder_class.edit_format == edit_format:
                res = coder_class(main_model, io, **kwargs) 
                res.original_kwargs = dict(kwargs) 
                return res
        
        valid_formats = [
            str(c.edit_format)
            for c in coders.__all__
            if hasattr(c, "edit_format") and c.edit_format is not None
        ]
        raise UnknownEditFormat(edit_format, valid_formats)

    def clone(self, **kwargs):
        new_coder = Coder.create(from_coder=self, **kwargs)
        return new_coder

    def get_announcements(self):
        lines = []
        lines.append(f"Aider v{__version__}")

        # Model
        main_model = self.main_model
        weak_model = main_model.weak_model

        if weak_model is not main_model:
            prefix = "Main model"
        else:
            prefix = "Model"

        output = f"{prefix}: {main_model.name} with {self.edit_format} edit format"

        # Check for thinking token budget
        thinking_tokens = main_model.get_thinking_tokens()
        if thinking_tokens:
            output += f", {thinking_tokens} think tokens"

        # Check for reasoning effort
        reasoning_effort = main_model.get_reasoning_effort()
        if reasoning_effort:
            output += f", reasoning {reasoning_effort}"

        if self.add_cache_headers or main_model.caches_by_default:
            output += ", prompt cache"
        if main_model.info.get("supports_assistant_prefill"):
            output += ", infinite output"

        lines.append(output)

        if self.edit_format == "architect":
            output = (
                f"Editor model: {main_model.editor_model.name} with"
                f" {main_model.editor_edit_format} edit format"
            )
            lines.append(output)

        if weak_model is not main_model:
            output = f"Weak model: {weak_model.name}"
            lines.append(output)

        # Repo
        if self.repo:
            rel_repo_dir = self.repo.get_rel_repo_dir()
            num_files = len(self.repo.get_tracked_files())

            lines.append(f"Git repo: {rel_repo_dir} with {num_files:,} files")
            if num_files > 1000:
                lines.append(
                    "Warning: For large repos, consider using --subtree-only and .aiderignore"
                )
                lines.append(f"See: {urls.large_repos}")
        else:
            lines.append("Git repo: none")

        # Repo-map
        if self.repo_map:
            map_tokens = self.repo_map.max_map_tokens
            if map_tokens > 0:
                refresh = self.repo_map.refresh
                lines.append(f"Repo-map: using {map_tokens} tokens, {refresh} refresh")
                max_map_tokens = self.main_model.get_repo_map_tokens() * 2
                if map_tokens > max_map_tokens:
                    lines.append(
                        f"Warning: map-tokens > {max_map_tokens} is not recommended. Too much"
                        " irrelevant code can confuse LLMs."
                    )
            else:
                lines.append("Repo-map: disabled because map_tokens == 0")
        else:
            lines.append("Repo-map: disabled")

        # Files
        
        if self.history_manager and self.history_manager.redis_client:
            lines.append(f"Chat history: Redis (Session ID: {self.history_manager.session_id})")
        else:
            lines.append("Chat history: In-memory (Redis not connected or configured)")

        for fname in self.get_inchat_relative_files(): lines.append(f"Added {fname} to the chat.")
        for fname_abs in self.abs_read_only_fnames: lines.append(f"Added {self.get_rel_fname(fname_abs)} to the chat (read-only).")
        if self.io.multiline_mode:
            lines.append("Multiline mode: Enabled. Enter inserts newline, Alt-Enter submits text")

        return lines

    ok_to_warm_cache = False

    def __init__(
        self,
        main_model,
        io,
        repo=None,
        fnames=None,
        read_only_fnames=None,
        show_diffs=False,
        auto_commits=True,
        dirty_commits=True,
        dry_run=False,
        map_tokens=1024,
        verbose=False,
        stream=True,
        use_git=True,
        # cur_messages, done_messages, restore_chat_history REMOVED
        history_manager=None, # ENSURED
        auto_lint=True,
        auto_test=False,
        lint_cmds=None,
        test_cmd=None,
        aider_commit_hashes=None,
        map_mul_no_files=8, # Default value
        commands=None,
        # summarizer REMOVED
        total_cost=0.0,
        analytics=None,
        map_refresh="auto",
        cache_prompts=False,
        num_cache_warming_pings=0,
        suggest_shell_commands=True,
        chat_language=None,
        detect_urls=True,
        ignore_mentions=None,
        file_watcher=None,
        auto_copy_context=False,
        auto_accept_architect=True,
        **kwargs, 
    ):
        self.analytics = analytics if analytics is not None else Analytics()
        self.event = self.analytics.event
        self.chat_language = chat_language
        self.aider_commit_hashes = set(aider_commit_hashes) if aider_commit_hashes else set()
        self.rejected_urls = set()
        self.abs_root_path_cache = {}
        self.auto_copy_context = auto_copy_context
        self.auto_accept_architect = auto_accept_architect
        self.ignore_mentions = ignore_mentions or set()
        self.file_watcher = file_watcher
        if self.file_watcher: self.file_watcher.coder = self
        self.suggest_shell_commands = suggest_shell_commands
        self.detect_urls = detect_urls
        self.num_cache_warming_pings = num_cache_warming_pings
        fnames = fnames or []
        self.io = io or InputOutput()
        self.chat_completion_call_hashes = []
        self.chat_completion_response_hashes = []
        self.need_commit_before_edits = set()
        self.total_cost = total_cost
        self.verbose = verbose
        self.abs_fnames = set()
        self.abs_read_only_fnames = set()
        self.history_manager = history_manager
        self.shell_commands = []
        if not auto_commits: dirty_commits = False
        self.auto_commits = auto_commits
        self.dirty_commits = dirty_commits
        self.dry_run = dry_run
        self.pretty = self.io.pretty
        self.main_model = main_model
        self.reasoning_tag_name = self.main_model.reasoning_tag or REASONING_TAG
        self.stream = stream and main_model.streaming
        if cache_prompts and self.main_model.cache_control: self.add_cache_headers = True
        self.show_diffs = show_diffs
        self.commands = commands or Commands(self.io, self)
        self.commands.coder = self
        self.repo = repo
        if use_git and self.repo is None:
            try:
                self.repo = GitRepo(self.io, fnames, None, models=main_model.commit_message_models())
            except FileNotFoundError: pass
        self.root = self.repo.root if self.repo else utils.find_common_root([str(Path(f).resolve()) for f in fnames]) or os.getcwd()
        
        for fname_arg in fnames:
            fname_path = Path(fname_arg)
            if not fname_path.is_absolute(): fname_path = Path(self.root) / fname_path # Resolve relative to determined root
            fname_path = fname_path.resolve()
            if self.repo and not str(fname_path).startswith(self.repo.root):
                self.io.tool_warning(f"File {fname_path} is outside git repo {self.repo.root}, skipping.")
                continue
            if self.repo and (self.repo.git_ignored_file(fname_path) or self.repo.ignored_file(fname_path)):
                self.io.tool_warning(f"Skipping {fname_path} due to gitignore/aiderignore.")
                continue
            if not fname_path.exists():
                if utils.touch_file(str(fname_path)): self.io.tool_output(f"Creating empty file {fname_path}")
                else: self.io.tool_warning(f"Cannot create {fname_path}, skipping."); continue
            if not fname_path.is_file():
                self.io.tool_warning(f"Skipping {fname_path} (not a normal file)."); continue
            abs_fname_str = str(fname_path)
            self.abs_fnames.add(abs_fname_str)
            if self.history_manager: self.history_manager.add_event(AddFileEvent(filepath=self.get_rel_fname(abs_fname_str)))
        self.check_added_files()

        if read_only_fnames:
            for ro_fname_arg in read_only_fnames:
                ro_fname_path = Path(ro_fname_arg)
                if not ro_fname_path.is_absolute(): ro_fname_path = Path(self.root) / ro_fname_path
                ro_fname_path = ro_fname_path.resolve()
                if ro_fname_path.exists() and ro_fname_path.is_file():
                    abs_ro_fname_str = str(ro_fname_path)
                    self.abs_read_only_fnames.add(abs_ro_fname_str)
                    if self.history_manager: self.history_manager.add_event(AddFileEvent(filepath=self.get_rel_fname(abs_ro_fname_str), read_only=True))
                else: self.io.tool_warning(f"Read-only file {ro_fname_path} not found/not a file, skipping.")
        
        use_repo_map = map_tokens > 0 if map_tokens is not None else main_model.use_repo_map
        if map_tokens is None: map_tokens = 1024 # Default if not specified and use_repo_map is true

        if use_repo_map and self.repo and hasattr(self, "gpt_prompts") and self.gpt_prompts.repo_content_prefix:
            self.repo_map = RepoMap(map_tokens, self.root, self.main_model, io, self.gpt_prompts.repo_content_prefix,
                                    self.verbose, self.main_model.info.get("max_input_tokens", 0),
                                    map_mul_no_files=map_mul_no_files, refresh=map_refresh)
        self.linter = Linter(root=self.root, encoding=io.encoding)
        self.auto_lint = auto_lint
        self.setup_lint_cmds(lint_cmds)
        self.auto_test = auto_test
        self.test_cmd = test_cmd
        if self.functions:
            from jsonschema import Draft7Validator
            for function in self.functions: Draft7Validator.check_schema(function)
            if self.verbose: self.io.tool_output(f"JSON Schema:\n{json.dumps(self.functions, indent=4)}")
        self.llmlingua_compressor = kwargs.pop("llmlingua_compressor", None)
        # ... (llmlingua setup logic remains) ...
        if kwargs: raise ValueError(f"Unexpected arguments: {kwargs}")

    def init_before_message(self):
        self.aider_edited_files = set()
        self.reflected_message = None
        self.num_reflections = 0
        self.lint_outcome = None
        self.test_outcome = None
        self.shell_commands = []
        self.message_cost = 0
        # commit_before_message REMOVED

    def check_added_files(self):
        """
        Perform any final checks or logging after files have been added to the chat.
        This method can be overridden by subclasses for specific behaviors.
        """
        # Placeholder implementation. Add specific checks if needed.
        pass

    def run_one(self, user_message, preproc):
        self.init_before_message()

        if preproc:
            message = self.preproc_user_input(user_message)
        else:
            message = user_message

        while message:
            self.reflected_message = None
            list(self.send_message(message)) # Process the generator

            if not self.reflected_message:
                break

            if self.num_reflections >= self.max_reflections:
                self.io.tool_warning(f"Only {self.max_reflections} reflections allowed, stopping.")
                return

            self.num_reflections += 1
            message = self.reflected_message

    # summarize_start, summarize_worker, summarize_end REMOVED
    # move_back_cur_messages REMOVED

    def format_chat_chunks(self):
        self.choose_fence()
        main_sys_content = self.fmt_system_prompt(self.gpt_prompts.main_system)
        if self.gpt_prompts.system_reminder:
            main_sys_content += "\n" + self.fmt_system_prompt(self.gpt_prompts.system_reminder)

        chunks = ChatChunks()
        if self.main_model.use_system_prompt:
            chunks.system = [dict(role="system", content=main_sys_content)]
        else:
            chunks.system = [dict(role="user", content=main_sys_content), dict(role="assistant", content="Ok.")]

        example_messages = []
        if self.gpt_prompts.example_messages:
            for msg_spec in self.gpt_prompts.example_messages:
                example_messages.append(dict(role=msg_spec["role"], content=self.fmt_system_prompt(msg_spec["content"])))
            example_messages.extend([
                dict(role="user", content="I switched to a new code base..."),
                dict(role="assistant", content="Ok.")
            ])
        chunks.examples = example_messages
        
        system_tokens = self.main_model.token_count(chunks.system)
        example_tokens = self.main_model.token_count(chunks.examples)
        
        chunks.repo = self.get_repo_messages()
        repo_map_tokens = self.main_model.token_count(chunks.repo) if chunks.repo else 0

        chunks.readonly_files = self.get_readonly_files_messages()
        readonly_files_tokens = self.main_model.token_count(chunks.readonly_files) if chunks.readonly_files else 0
        
        chunks.chat_files = self.get_chat_files_messages()
        chat_files_tokens = self.main_model.token_count(chunks.chat_files) if chunks.chat_files else 0

        max_context_tokens = self.main_model.info.get("max_input_tokens", 8192)
        
        history_max_tokens_from_args = 4096 
        if hasattr(self.io, 'args') and self.io.args and hasattr(self.io.args, 'history_max_tokens'):
            history_max_tokens_from_args = self.io.args.history_max_tokens or 4096

        other_context_tokens = system_tokens + example_tokens + repo_map_tokens + readonly_files_tokens + chat_files_tokens
        
        # Calculate available tokens for history, considering a potential reminder later
        # For now, don't subtract reminder tokens yet, as it's conditional
        available_for_history = max_context_tokens - other_context_tokens
        
        history_token_limit = min(available_for_history, history_max_tokens_from_args)
        history_token_limit = max(history_token_limit, 100) 

        if self.history_manager:
            chunks.history = self.history_manager.generate_llm_context(
                max_tokens=history_token_limit,
                tokenizer_func=self.main_model.token_count,
                compaction_level=1 # Enable first level of compaction
            )
        else:
            chunks.history = []

        reminder_message_list = []
        if self.gpt_prompts.system_reminder:
            reminder_message_list = [dict(role="system", content=self.fmt_system_prompt(self.gpt_prompts.system_reminder))]
        
        actual_reminder_tokens = self.main_model.token_count(reminder_message_list) if reminder_message_list else 0
        current_history_tokens = self.main_model.token_count(chunks.history)

        # Check if reminder fits with the *actual* history tokens used
        if self.gpt_prompts.system_reminder and (other_context_tokens + current_history_tokens + actual_reminder_tokens <= max_context_tokens):
            if self.main_model.reminder == "sys":
                chunks.reminder = reminder_message_list
        else:
            chunks.reminder = []
            if self.gpt_prompts.system_reminder and (other_context_tokens + current_history_tokens + actual_reminder_tokens > max_context_tokens):
                 # If reminder didn't fit, we might need to re-fetch history with less tokens to make space.
                 # This is complex. For now, if reminder doesn't fit, it's omitted.
                 # A more advanced approach would be to reserve space for reminder when fetching history.
                 if self.verbose: self.io.tool_output("System reminder omitted due to token limits after history retrieval.")


        # cur_messages REMOVED
        return chunks

    def send_message(self, inp):
        self.event("message_send_starting")
        self.io.llm_started()

        if self.history_manager:
            self.history_manager.add_event(UserPromptEvent(content=inp))

        messages = self.format_messages().all_messages() # Ensure all_messages() is called

        if not self.check_tokens(messages):
            self.io.llm_finished()
            return
        
        self.warm_cache(self.format_chat_chunks())

        if self.verbose: utils.show_messages(messages, functions=self.functions)
        self.multi_response_content = ""
        self.mdstream = self.io.get_assistant_mdstream() if self.show_pretty() and self.stream else None
        
        retry_delay = 0.125
        litellm_ex = LiteLLMExceptions()
        self.usage_report = None
        exhausted = False
        interrupted = False

        # LLMLingua block can remain as is for potential future use

        try:
            while True:
                try:
                    yield from self.send(messages, functions=self.functions) # Call the lower-level send
                    break
                except litellm_ex.exceptions_tuple() as err:
                    ex_info = litellm_ex.get_ex_info(err)
                    if ex_info.name == "ContextWindowExceededError": exhausted = True; break
                    should_retry = ex_info.retry
                    if should_retry:
                        retry_delay *= 2
                        if retry_delay > RETRY_TIMEOUT: should_retry = False
                    if not should_retry: self.mdstream = None; self.check_and_open_urls(err, ex_info.description); break
                    err_msg = str(err)
                    if ex_info.description: self.io.tool_warning(err_msg); self.io.tool_error(ex_info.description)
                    else: self.io.tool_error(err_msg)
                    self.io.tool_output(f"Retrying in {retry_delay:.1f} seconds..."); time.sleep(retry_delay)
                    continue
                except KeyboardInterrupt: interrupted = True; break
                except FinishReasonLength:
                    if not self.main_model.info.get("supports_assistant_prefill"): exhausted = True; break
                    self.multi_response_content = self.get_multi_response_content_in_progress()
                    if messages[-1]["role"] == "assistant": messages[-1]["content"] = self.multi_response_content
                    else: messages.append(dict(role="assistant", content=self.multi_response_content, prefix=True))
                except Exception as err:
                    self.mdstream = None
                    self.io.tool_warning("".join(traceback.format_exception(type(err), err, err.__traceback__)))
                    self.io.tool_error(str(err))
                    self.event("message_send_exception", exception=str(err))
                    return
        finally:
            if self.mdstream: self.live_incremental_response(True); self.mdstream = None
            self.partial_response_content = self.get_multi_response_content_in_progress(True)
            self.remove_reasoning_content() # This modifies self.partial_response_content
            
            if self.history_manager: # Log assistant reply here, after reasoning is removed
                self.add_assistant_reply_to_history() 
            
            self.multi_response_content = "" # Clear after logging
            self.io.llm_finished()

        self.io.tool_output()
        self.show_usage_report()

        if exhausted:
            self.show_exhausted_error()
            self.num_exhausted_context_windows += 1
            # No specific EditOutcomeEvent for this in provided history file
            return

        content_from_llm = self.partial_response_function_call.get("explanation", "") if self.partial_response_function_call else self.partial_response_content

        if not interrupted:
            add_rel_files_message = self.check_for_file_mentions(content_from_llm)
            if add_rel_files_message:
                self.reflected_message = (self.reflected_message + "\n\n" + add_rel_files_message) if self.reflected_message else add_rel_files_message
                # UserPromptEvent for reflection will be logged at start of next send_message call
                return
            try:
                if self.reply_completed(): pass # Reply logged in finally block
            except KeyboardInterrupt: interrupted = True

        if interrupted:
            # AssistantMessageEvent for interruption already logged by add_assistant_reply_to_history if content was "Interrupted..."
            # Or, if add_assistant_reply_to_history didn't run due to early exit, log here.
            # For simplicity, assume add_assistant_reply_to_history handles it or partial_response_content is empty.
            return

        edited_paths = self.apply_updates() # Logs ApplyEditEvent or UserRejectEditEvent

        if edited_paths:
            self.aider_edited_files.update(edited_paths)
            self.auto_commit(edited_paths) # Logs UserCommitEvent

        if self.reflected_message: return # Reflection will be new UserPromptEvent

        if edited_paths and self.auto_lint:
            lint_errors = self.lint_edited(edited_paths)
            self.auto_commit(edited_paths, context="Ran the linter") # Logs UserCommitEvent
            self.lint_outcome = not lint_errors
            if lint_errors:
                if self.io.confirm_ask("Attempt to fix lint errors?"):
                    self.reflected_message = lint_errors
                    # Reflection will be new UserPromptEvent
                    return
                # else: No specific EditOutcomeEvent for "lint_errors_not_fixed"

        self.run_shell_commands() # Logs RunCommandEvent and CommandOutputEvent

        if edited_paths and self.auto_test:
            test_errors = self.commands.cmd_test(self.test_cmd) # Logs events via cmd_test
            self.test_outcome = not test_errors
            if test_errors:
                if self.io.confirm_ask("Attempt to fix test errors?"):
                    self.reflected_message = test_errors
                    # Reflection will be new UserPromptEvent
                    return
                # else: No specific EditOutcomeEvent for "test_errors_not_fixed"

    def add_assistant_reply_to_history(self):
        if not self.history_manager: return

        content_to_log = self.partial_response_content # This has reasoning removed by now
        
        # Check for function call (this logic might need refinement based on how function calls are structured)
        # The provided history file does not have LLMFunctionCallEvent.
        # For now, if it's a function call, we'll log it as a special AssistantMessage.
        if self.partial_response_function_call:
            func_name = self.partial_response_function_call.get("name", "unknown_function")
            func_args_str = self.partial_response_function_call.get("arguments", "{}")
            # Create a string representation for the log
            log_content = f"[FUNCTION_CALL]\nName: {func_name}\nArguments: {func_args_str}"
            if content_to_log: # If there was also textual explanation
                log_content = f"{content_to_log}\n{log_content}"
            self.history_manager.add_event(AssistantMessageEvent(content=log_content))
        elif content_to_log: # Standard text or edit response
            # Determine if it's an edit based on format (simplified)
            is_edit = False
            if self.edit_format and self.edit_format != "text":
                if "```" in content_to_log or "--- a/" in content_to_log or "<source" in content_to_log:
                    is_edit = True
            
            if is_edit:
                self.history_manager.add_event(LLMResponseEditEvent(edit_content=content_to_log))
            else:
                self.history_manager.add_event(AssistantMessageEvent(content=content_to_log))
        # If both are empty, perhaps an empty response or only interruption, do nothing here.

    # send method (lower level) remains largely the same.
    # apply_updates method:
    def apply_updates(self):
        edited_paths_applied = set()
        try:
            edits = self.get_edits()
            edits_prepared = self.apply_edits_dry_run(edits)
            
            edits_to_apply = self.prepare_to_edit(edits_prepared) 
            if not edits_to_apply and edits_prepared: 
                if self.history_manager:
                    self.history_manager.add_event(UserRejectEditEvent()) # No reason field
                return edited_paths_applied

            self.apply_edits(edits_to_apply) 

            if not self.dry_run and edits_to_apply:
                current_edit_paths = set()
                for edit_op in edits_to_apply:
                    path = None
                    if isinstance(edit_op, dict) and 'fname' in edit_op: path = edit_op['fname']
                    elif hasattr(edit_op, 'fname'): path = edit_op.fname
                    elif isinstance(edit_op, str): path = edit_op # If edit_op is just a path string
                    
                    if path: current_edit_paths.add(self.get_rel_fname(path))

                if current_edit_paths and self.history_manager:
                    commit_hash = self.last_aider_commit_hash 
                    self.history_manager.add_event(ApplyEditEvent(
                        filepaths=list(current_edit_paths),
                        commit_hash=commit_hash 
                    ))
                edited_paths_applied.update(current_edit_paths)

        except ValueError as err: 
            self.num_malformed_responses += 1
            err_msg = str(err.args[0]) if err.args else "Unknown malformed response"
            self.io.tool_error("The LLM did not conform to the edit format.")
            self.io.tool_output(urls.edit_errors); self.io.tool_output(); self.io.tool_output(err_msg)
            self.reflected_message = err_msg 
            # No specific EditOutcomeEvent for malformed_response
            if self.history_manager: # Log as an assistant message or a generic error event if available
                self.history_manager.add_event(AssistantMessageEvent(content=f"Error: LLM response malformed. Details: {err_msg}"))
            return edited_paths_applied 
        except ANY_GIT_ERROR as err: 
            self.io.tool_error(str(err))
            # No specific EditOutcomeEvent for git_error
            if self.history_manager:
                self.history_manager.add_event(AssistantMessageEvent(content=f"Error: Git operation failed. Details: {str(err)}"))
            return edited_paths_applied
        except Exception as err: 
            self.io.tool_error("Exception while updating files:"); self.io.tool_error(str(err), strip=False); traceback.print_exc()
            self.reflected_message = str(err)
            # No specific EditOutcomeEvent for exception_applying_edits
            if self.history_manager:
                self.history_manager.add_event(AssistantMessageEvent(content=f"Error: Exception during file updates. Details: {str(err)}"))
            return edited_paths_applied

        for path_rel in edited_paths_applied: 
            self.io.tool_output(f"Applied edit to {path_rel}" if not self.dry_run else f"Did not apply edit to {path_rel} (--dry-run)")
        return edited_paths_applied

    # get_context_from_history REMOVED

    def auto_commit(self, edited_paths, context=None):
        if not self.repo or not self.auto_commits or self.dry_run or not edited_paths: return

        commit_context_msg = context or "Aider auto-commit" 
        # Try to get a more specific context from recent history if no explicit context is passed
        if not context and self.history_manager and self.history_manager.redis_client:
            try:
                last_events_texts = self.history_manager.redis_client.lrange(self.history_manager.text_list_key, -3, -1) or []
                if last_events_texts:
                    commit_context_msg = "Context:\n" + "\n".join(last_events_texts)
            except Exception as e: self.io.tool_warning(f"Failed to get history for commit context: {e}")
        
        try:
            relative_edited_paths = [self.get_rel_fname(p) for p in edited_paths]
            res = self.repo.commit(fnames=relative_edited_paths, context=commit_context_msg, aider_edits=True)
            if res:
                self.show_auto_commit_outcome(res)
                commit_hash, commit_message = res
                self.last_aider_commit_hash = commit_hash 
                self.aider_commit_hashes.add(commit_hash)
                if self.history_manager:
                    # Embed hash in message due to dataclass limitation
                    log_message = f"{commit_message} (hash: {commit_hash})"
                    if context: log_message += f" (context: {context})"
                    self.history_manager.add_event(UserCommitEvent(message=log_message))
                return self.gpt_prompts.files_content_gpt_edits.format(hash=commit_hash, message=commit_message)
        except ANY_GIT_ERROR as err:
            self.io.tool_error(f"Unable to auto-commit: {str(err)}")
            # No specific EditOutcomeEvent for auto_commit_failed
            if self.history_manager:
                self.history_manager.add_event(AssistantMessageEvent(content=f"Error: Auto-commit failed. Details: {str(err)}"))
        return None

    def show_undo_hint(self):
        if self.aider_commit_hashes: # Check if any aider commits were made
             self.io.tool_output("You can use /undo to undo the last aider commit.")

    def dirty_commit(self):
        if not self.need_commit_before_edits: return True # Nothing dirty that aider cares about
        if not self.dirty_commits:
            if self.history_manager: self.history_manager.add_event(UserRejectEditEvent()) # No reason field
            return False 
        if not self.repo: return True

        dirty_files_str = ", ".join(self.get_rel_fname(f) for f in self.need_commit_before_edits)
        if not self.io.confirm_ask(f"Commit uncommitted changes in: {dirty_files_str}?", default="y"):
            if self.history_manager: self.history_manager.add_event(UserRejectEditEvent()) # No reason field
            return False 

        relative_dirty_paths = [self.get_rel_fname(p) for p in self.need_commit_before_edits]
        res = self.repo.commit(fnames=relative_dirty_paths, context="Pre-aider-edit commit of user changes.")
        if res and self.history_manager:
             commit_hash, commit_message = res
             log_message = f"{commit_message} (hash: {commit_hash}, context: Pre-edit commit for {', '.join(relative_dirty_paths)})"
             self.history_manager.add_event(UserCommitEvent(message=log_message))
             self.aider_commit_hashes.add(commit_hash)
        self.need_commit_before_edits = set() 
        return True

    def run_shell_commands(self): # Logs events via handle_shell_commands
        if not self.suggest_shell_commands or not self.shell_commands: return ""
        done = set()
        group = ConfirmGroup(set(self.shell_commands))
        accumulated_output = ""
        for command_block in self.shell_commands:
            if command_block in done: continue
            done.add(command_block)
            output_this_block, _ = self.handle_shell_commands(command_block, group)
            if output_this_block is not None: accumulated_output += output_this_block + "\n\n"
        self.shell_commands = []
        return accumulated_output.strip()

    def handle_shell_commands(self, commands_str_block, group):
        commands_to_run = [cmd.strip() for cmd in commands_str_block.strip().splitlines() if cmd.strip() and not cmd.strip().startswith("#")]
        if not commands_to_run: return None, -1
        
        prompt_q = "Run shell command?" if len(commands_to_run) == 1 else "Run shell commands?"
        if not self.io.confirm_ask(prompt_q, subject="\n".join(commands_to_run), explicit_yes_required=True, group=group, allow_never=True):
            if self.history_manager:
                for cmd in commands_to_run: self.history_manager.add_event(RunCommandEvent(command=cmd)) # No outcome field
            return None, -1 
        
        block_output = ""
        last_status = 0
        for command in commands_to_run:
            if self.history_manager: self.history_manager.add_event(RunCommandEvent(command=command)) # No outcome field
            self.io.tool_output(f"\nRunning: {command}")
            self.io.add_to_input_history(f"/run {command}")
            cwd = self.root if hasattr(self, 'root') and self.root else None
            status, output = run_cmd(command, error_print=self.io.tool_error, cwd=cwd)
            last_status = status
            
            current_cmd_output = f"Output from `{command}`:\n{output if output is not None else ''}\n"
            block_output += current_cmd_output

            if self.history_manager:
                log_out = output if output is not None else ""
                # Truncate for log
                max_log_len = 2048
                if len(log_out) > max_log_len:
                    log_out = log_out[:max_log_len//2] + "\n...\n" + log_out[-max_log_len//2:]
                self.history_manager.add_event(CommandOutputEvent(command=command, output=log_out, exit_status=status))
                self.io.tool_output(f"Logged output from `{command}` (status: {status}).")
        return block_output, last_status

    # --- Placeholder methods that need concrete implementation in subclasses ---
    def get_edits(self):
        """Placeholder: Subclasses must implement this to parse edits from LLM response."""
        raise NotImplementedError

    def apply_edits(self, edits):
        """Placeholder: Subclasses must implement this to apply edits to files."""
        raise NotImplementedError

    def apply_edits_dry_run(self, edits):
        """Placeholder: Subclasses must implement this for dry-run edit application."""
        # Default implementation: return edits unchanged if no dry-run logic needed
        return edits

    def choose_fence(self):
        """Placeholder: Subclasses might need to implement fence selection logic."""
        # Default: Use standard triple backticks
        self.fence = "```"

    def fmt_system_prompt(self, prompt):
        """Placeholder: Subclasses might format system prompts differently."""
        return prompt # Default: return as is

    def get_repo_messages(self):
        """Placeholder: Subclasses might format repo context differently."""
        if self.repo_map:
            repo_content = self.repo_map.get_repo_map(self.get_inchat_relative_files(), self.abs_read_only_fnames)
            if repo_content:
                return [dict(role="user", content=repo_content)]
        return []

    def get_readonly_files_messages(self):
        """Placeholder: Subclasses might format read-only files differently."""
        # Default implementation (similar to original logic)
        msgs = []
        for fname in self.abs_read_only_fnames:
            content = self.io.read_text(fname)
            if content is None:
                continue
            rel_fname = self.get_rel_fname(fname)
            content = f"{rel_fname}\n```\n{content}\n```\n"
            msgs.append(dict(role="user", content=content))
        return msgs

    def get_chat_files_messages(self):
        """Placeholder: Subclasses might format chat files differently."""
        # Default implementation (similar to original logic)
        msgs = []
        for fname in self.abs_fnames:
            content = self.io.read_text(fname)
            if content is None:
                continue
            rel_fname = self.get_rel_fname(fname)
            content = f"{rel_fname}\n```\n{content}\n```\n"
            msgs.append(dict(role="user", content=content))
        return msgs

    def check_tokens(self, messages):
        """Placeholder: Subclasses might have specific token checks."""
        # Default implementation (similar to original logic)
        try:
            tokens = self.main_model.token_count(messages)
        except Exception as e:
            self.io.tool_error(f"Failed to count tokens: {e}")
            return False

        if not tokens:
            return False

        max_tokens = self.main_model.info.get("max_input_tokens")
        if not max_tokens:
            return True

        if tokens > max_tokens:
            self.io.tool_error(f"Input is too large: {tokens} tokens, max is {max_tokens}")
            return False

        return True

    def warm_cache(self, chunks):
        """Placeholder: Subclasses might implement cache warming differently."""
        # Default implementation (similar to original logic)
        if not self.ok_to_warm_cache:
            return
        if not self.add_cache_headers:
            return
        if not self.num_cache_warming_pings:
            return

        self.ok_to_warm_cache = False

        if self.cache_warming_thread and self.cache_warming_thread.is_alive():
            return

        self.cache_warming_thread = threading.Thread(
            target=self.cache_warming_worker, args=(chunks,)
        )
        self.cache_warming_thread.daemon = True
        self.cache_warming_thread.start()

    def cache_warming_worker(self, chunks):
        """Placeholder: Subclasses might implement cache warming worker differently."""
        # Default implementation (similar to original logic)
        for i in range(self.num_cache_warming_pings):
            chunks.add_cache_control_headers()
            messages = chunks.all_messages()
            try:
                _, completion = self.main_model.send_completion(
                    messages,
                    None,
                    False,
                    self.temperature,
                )
            except Exception:
                return

    def preproc_user_input(self, inp):
        """Placeholder: Subclasses might preprocess user input differently."""
        return inp # Default: return as is

    def reply_completed(self):
        """Placeholder: Subclasses determine completion based on response."""
        # Default: Assume completion if there's content or function call
        return bool(self.partial_response_content or self.partial_response_function_call)

    def keyboard_interrupt(self):
        """Placeholder: Subclasses might handle interrupts differently."""
        self.last_keyboard_interrupt = time.time()

    def check_and_open_urls(self, err, description):
        """Placeholder: Subclasses might handle URL opening differently."""
        # Default implementation (similar to original logic)
        if not self.detect_urls:
            self.io.tool_error(str(err))
            if description:
                self.io.tool_error(description)
            return

        urls_to_open = utils.find_urls(str(err))
        if description:
            urls_to_open += utils.find_urls(description)

        urls_to_open = [url for url in urls_to_open if url not in self.rejected_urls]

        if not urls_to_open:
            self.io.tool_error(str(err))
            if description:
                self.io.tool_error(description)
            return

        self.io.tool_error(str(err))
        if description:
            self.io.tool_error(description)

        for url in urls_to_open:
            if self.io.confirm_ask(f"Open URL {url} in browser?"):
                utils.open_urls([url])
            else:
                self.rejected_urls.add(url)

    def abs_root_path(self, path):
        """Placeholder: Subclasses might handle path resolution differently."""
        # Default implementation (similar to original logic)
        if path in self.abs_root_path_cache:
            return self.abs_root_path_cache[path]

        res = Path(path)
        if not res.is_absolute():
            res = Path(self.root) / res

        res = res.resolve()
        self.abs_root_path_cache[path] = str(res)
        return str(res)

    def show_pretty(self):
        """Placeholder: Subclasses might determine prettiness differently."""
        return self.pretty # Default: use io setting

    def lint_edited(self, edited):
        """Placeholder: Subclasses might handle linting differently."""
        # Default implementation (similar to original logic)
        if not self.linter:
            return None

        edited_rel = [self.get_rel_fname(f) for f in edited]
        lint_errors = self.linter.lint(edited_rel)
        if lint_errors:
            self.io.tool_output(lint_errors)
            return lint_errors
        else:
            self.io.tool_output("No lint errors found.")
            return None

    def show_exhausted_error(self):
        """Placeholder: Subclasses might show exhaustion errors differently."""
        # Default implementation (similar to original logic)
        self.io.tool_error(
            "The conversation history is too long for the configured context window."
        )
        self.io.tool_error(
            f"Try running with a larger model, or with --history-max-tokens {self.main_model.info.get('max_input_tokens', 8192)//2} or smaller."
        )
        # Removed summarization hint
        # self.io.tool_error("Or use /clear to clear the history.")
