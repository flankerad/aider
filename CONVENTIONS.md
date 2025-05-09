# Global AI Conventions

## Core Interaction:
- **Concise & Token-Efficient:** Use minimal tokens. Be clear.
- **Truthful & Aware:** State knowledge limits. Admit if unable to fulfill.
- **Task-Focused:** No topic drift.
- **Independent & Proactive:** Warn of issues/risks; suggest improvements.
- **Goal Review:** Every 10 exchanges, confirm goal alignment.

## General Development Philosophy & Code Craftsmanship:
- **Pragmatic Solutions:** Deployable & functional over theoretical perfection.
- **Work First, Then Optimize:** 1. Make it work to spec. 2. Then, optimize/refactor.
- **Simplicity & No Overengineering:** Favor simple implementations. Avoid premature complexity.
- **Core Values:** Design for modular, reusable components.
- **SOLID:** Guidelines, not dogma. Be pragmatic.
- **Meaningful Names:** Variables, functions, etc., must have names that clearly reveal intent. Code should be self-explanatory.
- **Function Design:**
    - **Small & Focused:** Functions should be very small (a few lines if possible), do one thing well.
    - **Few Arguments:** Prefer fewer arguments (ideally 2-3 max).
- **Commenting:**
    - Strive for self-explanatory code; minimize comments.
    - If necessary, comments must add significant value not obvious from the code and avoid merely restating logic.
- **Error Handling:** Use exceptions for error handling, not error codes.
- **Secure Coding:** Actively consider and implement security best practices.
- **Existing Projects:** Minimal, non-intrusive changes. Prefer new, independent, plug-and-play modules/folders for new features.
- **New Projects:** Feature-based scaffolding; roadmap-driven.

- **Awareness & Goal:** Be aware of `task-tracker.md`. Goal is to help keep its `TASK_STATUS` array and summary counts (`META.completed_tasks`, `PRIORITY_SUMMARY`) accurate.
- **Suggest Task Updates:** For tasks in `TASK_STATUS` (identified by `id`):
    - As work starts/progresses/completes, suggest changes to:
        - `status`: PENDING -> IN_PROGRESS -> DONE
        - `started` (ISO8601 timestamp for IN_PROGRESS)
        - `completed` (ISO8601 timestamp for DONE)
        - `progress` (numeric, e.g., 0, 10, 50, 100)
        - `notes` (brief, factual updates).
    - Upon task completion, also suggest updates to `META.completed_tasks` and `PRIORITY_SUMMARY`.
- **Mode-Specific Triggers for Suggestions:**
    - **Architect:** If plans alter a task's scope/details in `task-tracker.md`, suggest relevant field/`notes` updates.
    - **Coder:**
        - On task start: Suggest `status: IN_PROGRESS`, `started` timestamp, `progress` (e.g., 10%).
        - On task completion: Suggest `status: DONE`, `completed` timestamp, `progress: 100%`.
- **Key Constraints:**
    - **JSON Integrity:** Ensure all suggested changes maintain `task-tracker.md`'s JSON structure and formatting.
    - **User Approval:** All suggestions require user review and commit. Your role is to prepare accurate update proposals.
