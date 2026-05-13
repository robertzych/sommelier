# Sommelier — Claude Code Instructions

## Development Conventions

- Each step that introduces new code with logic must also include unit tests before the step is considered complete and moved to Completed Steps in plan.md.
- After each step is completed (code written, tests passing, plan.md updated), commit and push the changes.
- When marking a step complete in plan.md: remove it from the Next Steps section and move it to the Completed Steps section. Update the entry with any significant findings, changes, or decisions made during implementation.
- Always use `uv run` to invoke Python tools (e.g., `uv run pytest`, `uv run python`). Do not use bare `python` or `python -m` directly.

## Design Decisions (Do Not Re-derive)

- **No `source_url` in chunk payloads**: GitBook URL mapping is unreliable without site settings access, so `source_url` was explicitly removed from chunk payloads in `ingest.py`. Do not add URL-generation logic (e.g., `_to_source_url`) in other modules — it will produce incorrect URLs. If source URLs are needed in future, the design decision must be revisited in `plan.md` first.
