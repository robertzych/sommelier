# Setup

## Environment Setup

**Prerequisites**: [git](https://git-scm.com/), [uv](https://github.com/astral-sh/uv), Python 3.11+, and an `ANTHROPIC_API_KEY` (or another [LiteLLM-supported](https://docs.litellm.ai/docs/providers) provider — see [configuration.md](configuration.md)).

Install `uv` if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**1. Clone and install**

```bash
git clone https://github.com/robertzych/sommelier.git
cd sommelier
uv sync
```

`uv sync` downloads all Python dependencies into an isolated virtual environment. Pre-built Qdrant data is included in the repo — no separate download needed.

**2. Configure**

```bash
cp sommelier.toml.example sommelier.toml
```

Edit `sommelier.toml` and set your API key:

```toml
[inference]
model = "anthropic/claude-haiku-4-5-20251001"
api_key = "sk-ant-..."   # or export ANTHROPIC_API_KEY in your shell
```

---

## Using Sommelier with Claude Code

**1. Complete [Environment Setup](#environment-setup) (steps 1–2)**

**2. Create `.mcp.json` from the example**

```bash
cp .mcp.json.example .mcp.json
```

Edit `.mcp.json` and set `--directory` to the absolute path of the cloned repo:

```json
{
  "mcpServers": {
    "sommelier": {
      "command": "uv",
      "args": ["run", "--directory", "/where/you/cloned/sommelier", "sommelier-mcp"]
    }
  }
}
```

Claude Code inherits `ANTHROPIC_API_KEY` from your shell, so no `env` block is needed. Open a Claude Code session in the `sommelier/` directory and ask any Apache Pinot question — the tool is called automatically. Follow-up questions carry full conversation context.

**Using Sommelier in other projects**: copy `.mcp.json.example` and `.claude/settings.json` to that project's root directory, rename to `.mcp.json`, and update `--directory` to point to your Sommelier clone.

---

## Using Sommelier with Claude Desktop

**1. Complete [Environment Setup](#environment-setup) (steps 1–2)**

**2. Edit Claude Desktop's config file**

Open `~/Library/Application Support/Claude/claude_desktop_config.json` (create it if it doesn't exist) and add the `sommelier` server:

```json
{
  "mcpServers": {
    "sommelier": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/where/you/cloned/sommelier",
        "sommelier-mcp"
      ],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

Update `--directory` to the absolute path of the cloned repo.

**3. Add a global instruction to Claude Desktop**

Go to **Settings > General > Instructions for Claude** and add:

> For any Apache Pinot question, always use the search_pinot tool.

Without this, Claude may answer well-known Pinot facts from training data rather than calling the tool. With it, no query prefix is needed — ask Pinot questions naturally and Sommelier is invoked automatically.

**4. Restart Claude Desktop and test**

Ask any Apache Pinot question. You should see a tool call to `search_pinot` and a response with a Sources section at the end.

---

## Using Sommelier from the CLI

**1. Complete [Environment Setup](#environment-setup) (steps 1–2)**

**One-shot query:**

```bash
uv run sommelier query "What is the default broker port?"
```

**Multi-turn REPL** (conversation memory across turns, Ctrl+C to exit):

```bash
uv run sommelier chat
```

**First run only**: FastEmbed downloads the embedding (219 MB) and reranker (92 MB) ONNX models on the first query. Subsequent runs use the cached models in `~/.cache`.