#!/usr/bin/env python

import base64
import datetime # Added
import hashlib
import json # Added
import locale
import math
import mimetypes
import os # Added
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
from typing import List, Optional # Added Optional

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
from aider.aider_history_redis import ( # Import new history manager and events
    RedisHistoryManager, UserPromptEvent, AssistantMessageEvent, AddFileEvent,
    DropFileEvent, LLMResponseEditEvent, ApplyEditEvent, UserRejectEditEvent,
    RunCommandEvent, CommandOutputEvent, UserCommitEvent, ModeChangeEvent,
    SettingChangeEvent, AddWebcontentEvent, PasteContentEvent, ClearHistoryEvent,
    ResetChatEvent
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
    # commit_before_message = [] # Removed, managed implicitly by history
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
    history_manager: Optional[RedisHistoryManager] = None # Added

    # llmlingua attributes (kept for potential future use)
    llmlingua_compressor = None
    compression_output_counter = 0
    compression_output_dir = "llmlingua_outputs"


    @classmethod
    def create(
        self,
        main_model=None,
        edit_format=None,
        io=None,
        from_coder=None,
        # summarize_from_coder=True, # Removed, no summarization
        history_manager=None, # Added
        **kwargs,
    ):
        import aider.coders as coders

        if not main_model:
            if from_coder:
                main_model = from_coder.main_model
            else:
                main_model = models.Model(models.DEFAULT_MODEL_NAME)

        if edit_format == "code":
            edit_format = None
        if edit_format is None:
            if from_coder:
                edit_format = from_coder.edit_format
            else:
                edit_format = main_model.edit_format

        if not io and from_coder:
            io = from_coder.io

        if not history_manager and from_coder: # Pass history manager when cloning
             history_manager = from_coder.history_manager

        if from_coder:
            use_kwargs = dict(from_coder.original_kwargs)  # copy orig kwargs

            # Bring along context from the old Coder
            update = dict(
                fnames=list(from_coder.abs_fnames),
                read_only_fnames=list(from_coder.abs_read_only_fnames),
                # done_messages=done_messages, # Removed
                # cur_messages=from_coder.cur_messages, # Removed
                aider_commit_hashes=from_coder.aider_commit_hashes,
                commands=from_coder.commands.clone(),
                total_cost=from_coder.total_cost,
                ignore_mentions=from_coder.ignore_mentions,
                file_watcher=from_coder.file_watcher,
                history_manager=history_manager, # Ensure it's passed
            )
            use_kwargs.update(update)  # override to complete the switch
            use_kwargs.update(kwargs)  # override passed kwargs

            kwargs = use_kwargs
            from_coder.ok_to_warm_cache = False

        # Ensure history_manager is passed down
        kwargs['history_manager'] = history_manager

        for coder_class in coders.__all__:
            if hasattr(coder_class, "edit_format") and coder_class.edit_format == edit_format:
                # Pass history_manager to the constructor
                res = coder_class(main_model, io, **kwargs)
                res.original_kwargs = dict(kwargs) # Store original kwargs *after* updates
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
        for fname in self.get_inchat_relative_files():
            lines.append(f"Added {fname} to the chat.")

        for fname in self.abs_read_only_fnames:
            rel_fname = self.get_rel_fname(fname)
            lines.append(f"Added {rel_fname} to the chat (read-only).")

        # Removed done_messages check
        # if self.done_messages:
        #     lines.append("Restored previous conversation history.")

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
        # cur_messages=None, # Removed
        # done_messages=None, # Removed
        # restore_chat_history=False, # Removed
        history_manager=None, # Added
        auto_lint=True,
        auto_test=False,
        lint_cmds=None,
        test_cmd=None,
        aider_commit_hashes=None,
        map_mul_no_files=8,
        commands=None,
        # summarizer=None, # Removed
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
        **kwargs, # Capture other potential args like llmlingua_compressor
    ):
        # Fill in a dummy Analytics if needed, but it is never .enable()'d
        self.analytics = analytics if analytics is not None else Analytics()

        self.event = self.analytics.event
        self.chat_language = chat_language
        # self.commit_before_message = [] # Removed
        self.aider_commit_hashes = set()
        self.rejected_urls = set()
        self.abs_root_path_cache = {}

        self.auto_copy_context = auto_copy_context
        self.auto_accept_architect = auto_accept_architect

        self.ignore_mentions = ignore_mentions
        if not self.ignore_mentions:
            self.ignore_mentions = set()

        self.file_watcher = file_watcher
        if self.file_watcher:
            self.file_watcher.coder = self

        self.suggest_shell_commands = suggest_shell_commands
        self.detect_urls = detect_urls

        self.num_cache_warming_pings = num_cache_warming_pings

        if not fnames:
            fnames = []

        if io is None:
            io = InputOutput()

        if aider_commit_hashes:
            self.aider_commit_hashes = aider_commit_hashes
        else:
            self.aider_commit_hashes = set()

        self.chat_completion_call_hashes = []
        self.chat_completion_response_hashes = []
        self.need_commit_before_edits = set()

        self.total_cost = total_cost

        self.verbose = verbose
        self.abs_fnames = set()
        self.abs_read_only_fnames = set()

        # Removed cur_messages/done_messages initialization
        # if cur_messages:
        #     self.cur_messages = cur_messages
        # else:
        #     self.cur_messages = []
        # if done_messages:
        #     self.done_messages = done_messages
        # else:
        #     self.done_messages = []

        self.io = io
        self.history_manager = history_manager # Store the history manager

        self.shell_commands = []

        if not auto_commits:
            dirty_commits = False

        self.auto_commits = auto_commits
        self.dirty_commits = dirty_commits

        self.dry_run = dry_run
        self.pretty = self.io.pretty

        self.main_model = main_model
        # Set the reasoning tag name based on model settings or default
        self.reasoning_tag_name = (
            self.main_model.reasoning_tag if self.main_model.reasoning_tag else REASONING_TAG
        )

        self.stream = stream and main_model.streaming

        if cache_prompts and self.main_model.cache_control:
            self.add_cache_headers = True

        self.show_diffs = show_diffs

        self.commands = commands or Commands(self.io, self)
        self.commands.coder = self

        self.repo = repo
        if use_git and self.repo is None:
            try:
                self.repo = GitRepo(
                    self.io,
                    fnames,
                    None,
                    models=main_model.commit_message_models(),
                )
            except FileNotFoundError:
                pass

        if self.repo:
            self.root = self.repo.root

        for fname in fnames:
            fname = Path(fname)
            if self.repo and self.repo.git_ignored_file(fname):
                self.io.tool_warning(f"Skipping {fname} that matches gitignore spec.")

            if self.repo and self.repo.ignored_file(fname):
                self.io.tool_warning(f"Skipping {fname} that matches aiderignore spec.")
                continue

            if not fname.exists():
                if utils.touch_file(fname):
                    self.io.tool_output(f"Creating empty file {fname}")
                else:
                    self.io.tool_warning(f"Can not create {fname}, skipping.")
                    continue

            if not fname.is_file():
                self.io.tool_warning(f"Skipping {fname} that is not a normal file.")
                continue

            fname = str(fname.resolve())

            self.abs_fnames.add(fname)
            self.check_added_files()

        if not self.repo:
            self.root = utils.find_common_root(self.abs_fnames)

        if read_only_fnames:
            self.abs_read_only_fnames = set()
            for fname in read_only_fnames:
                abs_fname = self.abs_root_path(fname)
                if os.path.exists(abs_fname):
                    self.abs_read_only_fnames.add(abs_fname)
                else:
                    self.io.tool_warning(f"Error: Read-only file {fname} does not exist. Skipping.")

        if map_tokens is None:
            use_repo_map = main_model.use_repo_map
            map_tokens = 1024
        else:
            use_repo_map = map_tokens > 0

        max_inp_tokens = self.main_model.info.get("max_input_tokens") or 0

        has_map_prompt = hasattr(self, "gpt_prompts") and self.gpt_prompts.repo_content_prefix

        if use_repo_map and self.repo and has_map_prompt:
            self.repo_map = RepoMap(
                map_tokens,
                self.root,
                self.main_model,
                io,
                self.gpt_prompts.repo_content_prefix,
                self.verbose,
                max_inp_tokens,
                map_mul_no_files=map_mul_no_files,
                refresh=map_refresh,
            )

        # Removed summarizer initialization
        # self.summarizer = summarizer or ChatSummary(
        #     [self.main_model.weak_model, self.main_model],
        #     self.main_model.max_chat_history_tokens,
        # )
        # self.summarizer_thread = None
        # self.summarized_done_messages = []
        # self.summarizing_messages = None

        # Removed history restoration from file
        # if not self.done_messages and restore_chat_history:
        #     history_md = self.io.read_text(self.io.chat_history_file)
        #     if history_md:
        #         self.done_messages = utils.split_chat_history_markdown(history_md)
        #         self.summarize_start()

        # Linting and testing
        self.linter = Linter(root=self.root, encoding=io.encoding)
        self.auto_lint = auto_lint
        self.setup_lint_cmds(lint_cmds) # Call the setup method
        # self.lint_cmds = lint_cmds # Removed redundant assignment
        self.auto_test = auto_test
        self.test_cmd = test_cmd

        # validate the functions jsonschema
        if self.functions:
            from jsonschema import Draft7Validator

            for function in self.functions:
                Draft7Validator.check_schema(function)

            if self.verbose:
                self.io.tool_output("JSON Schema:")
                self.io.tool_output(json.dumps(self.functions, indent=4))

        # Handle llmlingua compressor initialization (kept for potential future use)
        self.llmlingua_compressor = kwargs.pop("llmlingua_compressor", None)
        if self.llmlingua_compressor and PromptCompressor:
            try:
                os.makedirs(self.compression_output_dir, exist_ok=True)
                self.io.tool_output(
                    f"LLMLingua logging enabled. Outputs will be saved to '{self.compression_output_dir}'"
                )
            except OSError as e:
                self.io.tool_error(
                    f"Warning: Could not create LLMLingua output directory '{self.compression_output_dir}': {e}"
                )
                self.llmlingua_compressor = None  # Disable if dir creation fails
        elif self.llmlingua_compressor:
            self.io.tool_error(
                "Warning: LLMLingua compressor provided, but llmlingua library not installed or failed to import."
            )
            self.llmlingua_compressor = None  # Ensure it's disabled

        if kwargs: # Check for remaining kwargs after popping llmlingua
            raise ValueError(f"Unexpected arguments: {kwargs}")


    # Added setup_lint_cmds method
    def setup_lint_cmds(self, lint_cmds):
        """Configures the linter object with the appropriate commands."""
        if not self.auto_lint or not self.linter:
            if self.linter:
                # Explicitly disable in linter if auto_lint is off
                self.linter.set_linter_commands(None)
            return

        # Configure the linter with provided commands (or None for default)
        # Assumes self.linter has a method like set_linter_commands
        if self.linter:
            self.linter.set_linter_commands(lint_cmds)


    # ... (rest of the methods, modifications needed below) ...

    def init_before_message(self):
        self.aider_edited_files = set()
        self.reflected_message = None
        self.num_reflections = 0
        self.lint_outcome = None
        self.test_outcome = None
        self.shell_commands = []
        self.message_cost = 0
        # Removed commit_before_message handling
        # if self.repo:
        #     self.commit_before_message.append(self.repo.get_head_commit_sha())

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

    # Removed summarize_start, summarize_worker, summarize_end methods

    # Removed move_back_cur_messages method (logic needs rethinking with event log)

    def format_chat_chunks(self):
        self.choose_fence()
        main_sys = self.fmt_system_prompt(self.gpt_prompts.main_system)

        example_messages = []
        # ... (example message handling remains the same) ...
        if self.gpt_prompts.example_messages:
            example_messages += [
                dict(
                    role="user",
                    content=(
                        "I switched to a new code base. Please don't consider the above files"
                        " or try to edit them any longer."
                    ),
                ),
                dict(role="assistant", content="Ok."),
            ]

        if self.gpt_prompts.system_reminder:
            main_sys += "\n" + self.fmt_system_prompt(self.gpt_prompts.system_reminder)

        chunks = ChatChunks()

        if self.main_model.use_system_prompt:
            chunks.system = [
                dict(role="system", content=main_sys),
            ]
        else:
            chunks.system = [
                dict(role="user", content=main_sys),
                dict(role="assistant", content="Ok."),
            ]

        chunks.examples = example_messages

        # --- Replace history generation with call to RedisHistoryManager ---
        # Removed: self.summarize_end()
        # Removed: chunks.done = self.done_messages

        # Calculate token budget for history
        # This is approximate - needs refinement based on actual prompt structure
        system_tokens = self.main_model.token_count(chunks.system)
        example_tokens = self.main_model.token_count(chunks.examples)
        repo_map_tokens = 0
        readonly_files_tokens = 0
        chat_files_tokens = 0

        chunks.repo = self.get_repo_messages()
        if chunks.repo:
            repo_map_tokens = self.main_model.token_count(chunks.repo)

        chunks.readonly_files = self.get_readonly_files_messages()
        if chunks.readonly_files:
             readonly_files_tokens = self.main_model.token_count(chunks.readonly_files)

        chunks.chat_files = self.get_chat_files_messages()
        if chunks.chat_files:
             chat_files_tokens = self.main_model.token_count(chunks.chat_files)

        # Placeholder for reminder tokens - add later if needed
        reminder_tokens = 0

        max_context_tokens = self.main_model.info.get("max_input_tokens") or 8192 # Default if unknown
        # Use history_max_tokens arg if provided, otherwise calculate based on model limit
        max_hist_tokens_arg = self.io.args.history_max_tokens if hasattr(self.io, 'args') and self.io.args else 4096 # Get from args if available, handle None case
        max_hist_tokens = max_hist_tokens_arg or (max_context_tokens // 2) # Default to half context if not set

        other_context_tokens = (
            system_tokens + example_tokens + repo_map_tokens +
            readonly_files_tokens + chat_files_tokens + reminder_tokens
        )
        available_history_tokens = max(max_context_tokens - other_context_tokens, 100) # Ensure some minimum
        history_token_limit = min(available_history_tokens, max_hist_tokens)

        if self.history_manager:
            chunks.history = self.history_manager.generate_llm_context(
                max_tokens=history_token_limit,
                tokenizer_func=self.main_model.token_count
            )
        else:
            chunks.history = [] # No history if manager isn't available
        # --- End History Generation Replacement ---


        if self.gpt_prompts.system_reminder:
            reminder_message = [
                dict(
                    role="system", content=self.fmt_system_prompt(self.gpt_prompts.system_reminder)
                ),
            ]
        else:
            reminder_message = []

        # Removed cur messages handling
        # chunks.cur = list(self.cur_messages)
        chunks.reminder = [] # Reminder logic might need adjustment based on history

        # Recalculate total tokens with actual history
        messages_tokens = self.main_model.token_count(chunks.all_messages_except_cur()) # Exclude cur for now
        reminder_tokens = self.main_model.token_count(reminder_message)
        # cur_tokens = self.main_model.token_count(chunks.cur) # Removed cur

        # if None not in (messages_tokens, reminder_tokens, cur_tokens):
        #     total_tokens = messages_tokens + reminder_tokens + cur_tokens
        if None not in (messages_tokens, reminder_tokens):
             total_tokens = messages_tokens + reminder_tokens
        else:
            # add the reminder anyway
            total_tokens = 0

        # Removed final message handling based on cur
        # if chunks.cur:
        #     final = chunks.cur[-1]
        # else:
        #     final = None
        final = None # Assume no cur messages for now

        max_input_tokens = self.main_model.info.get("max_input_tokens") or 0
        # Add the reminder prompt if we still have room to include it.
        if (
            not max_input_tokens
            or total_tokens < max_input_tokens
            and self.gpt_prompts.system_reminder
        ):
            if self.main_model.reminder == "sys":
                chunks.reminder = reminder_message
            # Removed reminder injection into user message
            # elif self.main_model.reminder == "user" and final and final["role"] == "user":
            #     # stuff it into the user message
            #     new_content = (
            #         final["content"]
            #         + "\n\n"
            #         + self.fmt_system_prompt(self.gpt_prompts.system_reminder)
            #     )
            #     chunks.cur[-1] = dict(role=final["role"], content=new_content)

        return chunks

    def format_messages(self):
        chunks = self.format_chat_chunks()
        if self.add_cache_headers:
            # This might need adjustment if history format changes caching behavior
            chunks.add_cache_control_headers()

        # Return all messages including history now managed by ChatChunks
        return chunks.all_messages()

    def send_message(self, inp):
        self.event("message_send_starting")

        # Notify IO that LLM processing is starting
        self.io.llm_started()

        # Log user input event BEFORE formatting messages
        if self.history_manager:
            self.history_manager.add_event(UserPromptEvent(content=inp))

        # Format messages including the new history context
        messages = self.format_messages()

        if not self.check_tokens(messages):
             # Notify IO that LLM processing is ending (early exit)
            self.io.llm_finished()
            return
        self.warm_cache(self.format_chat_chunks()) # Pass chunks to warm_cache

        if self.verbose:
            utils.show_messages(messages, functions=self.functions)

        self.multi_response_content = ""
        if self.show_pretty() and self.stream:
            self.mdstream = self.io.get_assistant_mdstream()
        else:
            self.mdstream = None

        retry_delay = 0.125

        litellm_ex = LiteLLMExceptions()

        self.usage_report = None
        exhausted = False
        interrupted = False

        # <<< START LLMLINGUA LOGGING BLOCK (kept for potential future use) >>>
        if self.llmlingua_compressor and PromptCompressor:
            try:
                # 1. Calculate original tokens
                original_tokens = self.main_model.token_count(messages)

                # 2. Format messages for llmlingua (adjust as needed)
                system_prompt = ""
                history = []
                question = ""
                # Use the *already formatted* messages list
                for i, msg in enumerate(messages):
                    role = msg.get("role", "user").lower()
                    content = msg.get("content", "")
                    if isinstance(content, list): # Handle potential multipart content
                        content = " ".join(p.get('text', '') for p in content if p.get('type') == 'text')

                    if role == "system":
                        system_prompt += content + "\n" # Concatenate system prompts
                    elif role == "user":
                        if i == len(messages) - 1: # Assume last message is the question
                            question = content
                        else:
                            history.append(f"USER: {content}")
                    elif role == "assistant":
                        history.append(f"ASSISTANT: {content}")

                context_to_compress = "\n".join(history)
                system_prompt = system_prompt.strip()

                self.io.tool_output(
                    f"Attempting LLMLingua compression for logging (Original tokens: {original_tokens})..."
                )

                # 3. Call llmlingua compressor
                compressed_result = self.llmlingua_compressor.compress_prompt(
                    context=[context_to_compress],
                    instruction=system_prompt,
                    question=question,
                    return_details=True,
                )

                # 4. Extract compressed content and stats
                compressed_content = compressed_result.get("compressed_prompt", str(compressed_result))
                llm_reported_orig_tokens = compressed_result.get("origin_tokens", "?")
                llm_reported_comp_tokens = compressed_result.get("compressed_tokens", "?")

                # 5. Calculate compressed tokens using the main model's tokenizer
                compressed_message_for_calc = []
                if system_prompt:
                    compressed_message_for_calc.append({"role": "system", "content": system_prompt})
                compressed_message_for_calc.append({"role": "user", "content": compressed_content})
                if question:
                     compressed_message_for_calc.append({"role": "user", "content": question})

                calculated_compressed_tokens = self.main_model.token_count(
                    compressed_message_for_calc
                )

                self.io.tool_output("LLMLingua compression logged.")
                if llm_reported_orig_tokens != "?":
                    self.io.tool_output(
                        f"  LLMLingua stats: Original={llm_reported_orig_tokens}, Compressed={llm_reported_comp_tokens}"
                    )
                self.io.tool_output(
                    f"  Calculated stats: Original={original_tokens}, Compressed={calculated_compressed_tokens}"
                )

                # 6. Write to file
                self.compression_output_counter += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(
                    self.compression_output_dir,
                    f"compressed_{timestamp}_{self.compression_output_counter}.log",
                )
                try:
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(f"### LLMLingua Compression Log: {filename} ###\n")
                        f.write(f"Timestamp: {timestamp}\n")
                        f.write("\n--- Compression Stats ---\n")
                        f.write(f"Original Tokens (calculated): {original_tokens}\n")
                        f.write(f"Compressed Tokens (calculated): {calculated_compressed_tokens}\n")
                        reduction = original_tokens - calculated_compressed_tokens
                        percent_reduction = (
                            (reduction / original_tokens * 100) if original_tokens > 0 else 0
                        )
                        f.write(f"Token Reduction: {reduction} ({percent_reduction:.2f}%)\n")
                        if llm_reported_orig_tokens != "?":
                            f.write(
                                f"LLMLingua Reported Original Tokens: {llm_reported_orig_tokens}\n"
                            )
                            f.write(
                                f"LLMLingua Reported Compressed Tokens: {llm_reported_comp_tokens}\n"
                            )

                        f.write("\n--- Original Messages (Sent to LLM) ---\n")
                        try:
                            f.write(json.dumps(messages, indent=2))
                        except Exception:
                            f.write(str(messages))
                        f.write("\n\n--- Context Sent to LLMLingua ---\n")
                        f.write(f"Instruction:\n{system_prompt}\n\n")
                        f.write(f"History Context:\n{context_to_compress}\n\n")
                        f.write(f"Question:\n{question}\n")
                        f.write("\n--- Compressed Output from LLMLingua ---\n")
                        f.write(compressed_content)
                    self.io.tool_output(f"Compressed output and stats saved to: {filename}")
                except OSError as e:
                    self.io.tool_error(f"Error writing LLMLingua log file '{filename}': {e}")
                except Exception as e:
                    self.io.tool_error(
                        f"Unexpected error writing LLMLingua log file '{filename}': {e}"
                    )

            except Exception as e:
                import traceback
                self.io.tool_error(f"LLMLingua compression logging failed: {e}")
                traceback.print_exc()
        # <<< END LLMLINGUA LOGGING BLOCK >>>


        # --- Main LLM call ---
        try:
            while True:
                try:
                    yield from self.send(messages, functions=self.functions)
                    break
                # ... (existing error handling for LLM call) ...
                except litellm_ex.exceptions_tuple() as err:
                    ex_info = litellm_ex.get_ex_info(err)

                    if ex_info.name == "ContextWindowExceededError":
                        exhausted = True
                        break

                    should_retry = ex_info.retry
                    if should_retry:
                        retry_delay *= 2
                        if retry_delay > RETRY_TIMEOUT:
                            should_retry = False

                    if not should_retry:
                        self.mdstream = None
                        self.check_and_open_urls(err, ex_info.description)
                        break

                    err_msg = str(err)
                    if ex_info.description:
                        self.io.tool_warning(err_msg)
                        self.io.tool_error(ex_info.description)
                    else:
                        self.io.tool_error(err_msg)

                    self.io.tool_output(f"Retrying in {retry_delay:.1f} seconds...")
                    time.sleep(retry_delay)
                    continue
                except KeyboardInterrupt:
                    interrupted = True
                    break
                except FinishReasonLength:
                    # We hit the output limit!
                    if not self.main_model.info.get("supports_assistant_prefill"):
                        exhausted = True
                        break

                    self.multi_response_content = self.get_multi_response_content_in_progress()

                    # This logic might need review - appending to messages list for retry
                    if messages[-1]["role"] == "assistant":
                        messages[-1]["content"] = self.multi_response_content
                    else:
                        messages.append(
                            dict(role="assistant", content=self.multi_response_content, prefix=True)
                        )
                except Exception as err:
                    self.mdstream = None
                    lines = traceback.format_exception(type(err), err, err.__traceback__)
                    self.io.tool_warning("".join(lines))
                    self.io.tool_error(str(err))
                    self.event("message_send_exception", exception=str(err))
                    return
        finally:
            if self.mdstream:
                self.live_incremental_response(True)
                self.mdstream = None

            self.partial_response_content = self.get_multi_response_content_in_progress(True)
            self.remove_reasoning_content()
            self.multi_response_content = ""

            # Notify IO that LLM processing is finished
            self.io.llm_finished()


        self.io.tool_output()

        self.show_usage_report()

        # Log assistant reply event AFTER processing
        if self.history_manager:
             self.add_assistant_reply_to_history() # Use helper


        if exhausted:
            # Removed cur_messages handling
            self.show_exhausted_error()
            self.num_exhausted_context_windows += 1
            return

        if self.partial_response_function_call:
            args = self.parse_partial_args()
            if args:
                content = args.get("explanation") or ""
            else:
                content = ""
        elif self.partial_response_content:
            content = self.partial_response_content
        else:
            content = ""

        if not interrupted:
            add_rel_files_message = self.check_for_file_mentions(content)
            if add_rel_files_message:
                # This reflection logic needs review - how does it interact with event log?
                # For now, just log the reflection prompt if it happens.
                reflection_prompt = add_rel_files_message
                if self.reflected_message:
                    reflection_prompt = self.reflected_message + "\n\n" + add_rel_files_message
                self.reflected_message = reflection_prompt
                # Log the reflection prompt as a user event? Or system?
                # if self.history_manager:
                #     self.history_manager.add_event(UserPromptEvent(content=self.reflected_message))
                return

            try:
                if self.reply_completed():
                    return
            except KeyboardInterrupt:
                interrupted = True

        if interrupted:
            # Log interruption event?
            # if self.history_manager:
            #    self.history_manager.add_event(UserInterruptionEvent()) # Need new event type
            # Removed cur_messages handling
            # Log a simple assistant message about interruption
            if self.history_manager:
                 self.history_manager.add_event(AssistantMessageEvent(content="Interrupted."))
            return

        edited = self.apply_updates() # This needs to log ApplyEditEvent internally now

        if edited:
            self.aider_edited_files.update(edited)
            # auto_commit needs to log UserCommitEvent if successful
            saved_message = self.auto_commit(edited)

            # Removed move_back_cur_messages
            # if not saved_message and hasattr(self.gpt_prompts, "files_content_gpt_edits_no_repo"):
            #     saved_message = self.gpt_prompts.files_content_gpt_edits_no_repo
            # self.move_back_cur_messages(saved_message)

        if self.reflected_message:
             # Log reflection prompt?
             # if self.history_manager:
             #    self.history_manager.add_event(UserPromptEvent(content=self.reflected_message))
            return

        if edited and self.auto_lint:
            lint_errors = self.lint_edited(edited)
            # auto_commit needs to log UserCommitEvent
            self.auto_commit(edited, context="Ran the linter")
            self.lint_outcome = not lint_errors
            if lint_errors:
                ok = self.io.confirm_ask("Attempt to fix lint errors?")
                if ok:
                    self.reflected_message = lint_errors
                    # Log reflection prompt?
                    # if self.history_manager:
                    #    self.history_manager.add_event(UserPromptEvent(content=self.reflected_message))

                    return

        shared_output = self.run_shell_commands() # This logs RunCommand/Output events internally
        # Removed adding output back to cur_messages
        # if shared_output:
        #     self.cur_messages += [
        #         dict(role="user", content=shared_output),
        #         dict(role="assistant", content="Ok"),
        #     ]

        if edited and self.auto_test:
            # cmd_test needs to log RunCommand/Output events internally
            test_errors = self.commands.cmd_test(self.test_cmd)
            self.test_outcome = not test_errors
            if test_errors:
                ok = self.io.confirm_ask("Attempt to fix test errors?")
                if ok:
                    self.reflected_message = test_errors
                    # Log reflection prompt?
                    # if self.history_manager:
                    #    self.history_manager.add_event(UserPromptEvent(content=self.reflected_message))
                    return

    # ... (reply_completed, show_exhausted_error, lint_edited remain mostly the same) ...

    def __del__(self):
        """Cleanup when the Coder object is destroyed."""
        self.ok_to_warm_cache = False

    def add_assistant_reply_to_history(self):
        """Helper to log assistant messages/edits to the history manager."""
        if not self.history_manager:
            return

        if self.partial_response_content:
            # Check if it looks like an edit block based on edit_format (simplistic check)
            is_edit = False
            if self.edit_format in ["diff", "udiff", "whole", "diff-fenced"]: # Add other formats
                 # Basic check for fences or diff markers
                 if "```diff" in self.partial_response_content or "--- a/" in self.partial_response_content or (hasattr(self, 'fence') and self.fence in self.partial_response_content):
                      is_edit = True

            if is_edit:
                 # TODO: Extract target filename if possible from the content
                 target_file = None # Placeholder
                 self.history_manager.add_event(LLMResponseEditEvent(
                      edit_content=self.partial_response_content,
                      target_filepath=target_file
                 ))
            else:
                 self.history_manager.add_event(AssistantMessageEvent(content=self.partial_response_content))

        # Add function call logging if needed (TBD)
        # if self.partial_response_function_call:
        #     self.history_manager.add_event(AssistantFunctionCallEvent(...))


    # Removed add_assistant_reply_to_cur_messages

    # ... (get_file_mentions, check_for_file_mentions remain mostly the same) ...

    def send(self, messages, model=None, functions=None):
        # ... (logging block for llmlingua remains here for potential future use) ...

        # --- Main LLM call ---
        self.got_reasoning_content = False
        self.ended_reasoning_content = False

        if not model:
            model = self.main_model

        self.partial_response_content = ""
        self.partial_response_function_call = dict()

        self.io.log_llm_history("TO LLM", format_messages(messages))

        completion = None
        try:
            hash_object, completion = model.send_completion(
                messages,
                functions,
                self.stream,
                self.temperature,
            )
            self.chat_completion_call_hashes.append(hash_object.hexdigest())

            if self.stream:
                yield from self.show_send_output_stream(completion)
            else:
                self.show_send_output(completion)

            # Calculate costs for successful responses
            self.calculate_and_show_tokens_and_cost(messages, completion)

        except LiteLLMExceptions().exceptions_tuple() as err:
            ex_info = LiteLLMExceptions().get_ex_info(err)
            if ex_info.name == "ContextWindowExceededError":
                # Still calculate costs for context window errors
                self.calculate_and_show_tokens_and_cost(messages, completion)
            raise
        except KeyboardInterrupt as kbi:
            self.keyboard_interrupt()
            raise kbi
        finally:
            self.io.log_llm_history(
                "LLM RESPONSE",
                format_content("ASSISTANT", self.partial_response_content),
            )

            if self.partial_response_content:
                self.io.ai_output(self.partial_response_content)
            elif self.partial_response_function_call:
                # TODO: push this into subclasses
                args = self.parse_partial_args()
                if args:
                    self.io.ai_output(json.dumps(args, indent=4))

    # ... (show_send_output, show_send_output_stream, live_incremental_response, render_incremental_response, remove_reasoning_content remain mostly the same) ...

    # ... (calculate_and_show_tokens_and_cost, show_usage_report remain mostly the same) ...

    # ... (get_multi_response_content_in_progress remains the same) ...

    # ... (get_rel_fname, get_inchat_relative_files, is_file_safe, get_all_relative_files, get_all_abs_files, get_addable_relative_files remain the same) ...

    # ... (check_for_dirty_commit, allowed_to_edit, check_added_files, prepare_to_edit remain mostly the same) ...

    def apply_updates(self):
        edited = set()
        try:
            edits = self.get_edits()
            edits = self.apply_edits_dry_run(edits)
            edits = self.prepare_to_edit(edits) # This calls dirty_commit if needed
            edited_paths = set(edit for edit in edits if edit is not None)

            self.apply_edits(edits) # Apply the actual changes

            # Log ApplyEditEvent *after* successful application
            if self.history_manager and edited_paths and not self.dry_run:
                 # Get commit hash *after* potential auto-commit within apply_edits/auto_commit
                 commit_hash = self.last_aider_commit_hash # Use the stored hash
                 self.history_manager.add_event(ApplyEditEvent(
                     filepaths=list(edited_paths),
                     commit_hash=commit_hash
                 ))
            edited = edited_paths # Return the paths that were actually edited

        except ValueError as err:
            self.num_malformed_responses += 1
            err_msg = err.args
            self.io.tool_error("The LLM did not conform to the edit format.")
            self.io.tool_output(urls.edit_errors)
            self.io.tool_output()
            self.io.tool_output(str(err_msg))
            self.reflected_message = str(err_msg)
            # Log error event?
            # if self.history_manager:
            #     self.history_manager.add_event(MalformedResponseEvent(error=str(err_msg)))
            return edited # Return paths that might have been prepared

        except ANY_GIT_ERROR as err:
            self.io.tool_error(str(err))
            # Log error event?
            return edited
        except Exception as err:
            self.io.tool_error("Exception while updating files:")
            self.io.tool_error(str(err), strip=False)
            traceback.print_exc()
            self.reflected_message = str(err)
            # Log error event?
            return edited

        for path in edited:
            if self.dry_run:
                self.io.tool_output(f"Did not apply edit to {path} (--dry-run)")
            else:
                self.io.tool_output(f"Applied edit to {path}")

        return edited


    # ... (parse_partial_args remains the same) ...

    # Removed get_context_from_history

    def auto_commit(self, edited, context=None):
        if not self.repo or not self.auto_commits or self.dry_run:
            return

        # Removed context generation from history
        # if not context:
        #     context = self.get_context_from_history(self.cur_messages) # Needs replacement

        # Placeholder for context - how should commit message context be derived now?
        # Maybe use the last N events from history manager? Or just the last user prompt?
        commit_context = "Aider commit" # Basic placeholder
        if self.history_manager and self.history_manager.redis_client: # Check redis client
             # Get last ~5 text events as context? Needs careful design.
             try:
                 last_events = self.history_manager.redis_client.lrange(self.history_manager.text_list_key, -5, -1) or []
                 commit_context = "\n".join(last_events)
             except Exception as e:
                 self.io.tool_warning(f"Could not retrieve recent history for commit message context: {e}")


        try:
            res = self.repo.commit(fnames=edited, context=commit_context, aider_edits=True)
            if res:
                self.show_auto_commit_outcome(res)
                commit_hash, commit_message = res
                self.last_aider_commit_hash = commit_hash # Store the hash
                # Log UserCommitEvent triggered by auto-commit? Or rely on ApplyEditEvent's hash?
                # For now, rely on ApplyEditEvent logging the hash.
                return self.gpt_prompts.files_content_gpt_edits.format(
                    hash=commit_hash,
                    message=commit_message,
                )

            return self.gpt_prompts.files_content_gpt_no_edits
        except ANY_GIT_ERROR as err:
            self.io.tool_error(f"Unable to commit: {str(err)}")
            return

    # ... (show_auto_commit_outcome remains the same) ...

    def show_undo_hint(self):
        # Removed commit_before_message check
        # if not self.commit_before_message:
        #     return
        # if self.commit_before_message[-1] != self.repo.get_head_commit_sha():
        #     self.io.tool_output("You can use /undo to undo and discard each aider commit.")
        # Simplified check: if any aider commits exist, show hint
        if self.aider_commit_hashes:
             self.io.tool_output("You can use /undo to undo the last aider commit.")


    def dirty_commit(self):
        if not self.need_commit_before_edits:
            return
        if not self.dirty_commits:
            return
        if not self.repo:
            return

        res = self.repo.commit(fnames=self.need_commit_before_edits)
        if res and self.history_manager:
             # Log this pre-emptive commit as a USER_COMMIT?
             commit_hash, commit_message = res
             self.history_manager.add_event(UserCommitEvent(message=f"Pre-edit commit for: {', '.join(self.need_commit_before_edits)}"))


        # Removed move_back_cur_messages
        # self.move_back_cur_messages(self.gpt_prompts.files_content_local_edits)
        return True

    # ... (get_edits, apply_edits, apply_edits_dry_run remain placeholders) ...

    # ... (run_shell_commands, handle_shell_commands remain mostly the same, but need to log events) ...
    def run_shell_commands(self):
        if not self.suggest_shell_commands:
            return ""

        done = set()
        group = ConfirmGroup(set(self.shell_commands))
        accumulated_output = ""
        for command in self.shell_commands:
            if command in done:
                continue
            done.add(command)
            output, exit_status = self.handle_shell_commands(command, group) # Modified to return status
            if output is not None: # Check if command was run
                accumulated_output += output + "\n\n"
                # Log command output event here, after potential user confirmation
                if self.history_manager and self.io.confirm_ask(
                     "Add command output to the chat history log?", allow_never=True
                ):
                     # Truncate output before logging
                     max_len = 1000
                     truncated_output = output
                     if len(output) > max_len:
                          truncated_output = output[:max_len//2] + "\n...\n" + output[-max_len//2:]

                     self.history_manager.add_event(CommandOutputEvent(
                          command=command,
                          output=truncated_output,
                          exit_status=exit_status
                     ))
                     num_lines = len(output.strip().splitlines())
                     line_plural = "line" if num_lines == 1 else "lines"
                     self.io.tool_output(f"Logged {num_lines} {line_plural} of output.")

        # Return accumulated output for potential immediate display/use, though it's not added back to context
        return accumulated_output

    def handle_shell_commands(self, commands_str, group):
        commands = commands_str.strip().splitlines()
        command_count = sum(
            1 for cmd in commands if cmd.strip() and not cmd.strip().startswith("#")
        )
        prompt = "Run shell command?" if command_count == 1 else "Run shell commands?"
        if not self.io.confirm_ask(
            prompt,
            subject="\n".join(commands),
            explicit_yes_required=True,
            group=group,
            allow_never=True,
        ):
            return None, -1 # Indicate command was not run

        accumulated_output = ""
        final_exit_status = 0
        for command in commands:
            command = command.strip()
            if not command or command.startswith("#"):
                continue

            # Log RUN_COMMAND event *before* running
            if self.history_manager:
                 self.history_manager.add_event(RunCommandEvent(command=command))

            self.io.tool_output()
            self.io.tool_output(f"Running {command}")
            # Add the command to input history
            self.io.add_to_input_history(f"/run {command.strip()}")
            # Ensure self.root is used for cwd
            cwd_path = self.root if hasattr(self, 'root') and self.root else None
            exit_status, output = run_cmd(command, error_print=self.io.tool_error, cwd=cwd_path)
            final_exit_status = exit_status # Keep last exit status
            if output:
                accumulated_output += f"Output from {command}\n{output}\n"

        # Return output and status, logging happens in run_shell_commands
        return accumulated_output, final_exit_status

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