# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Using uv (recommended)
uv run streamlit run app.py

# Using pip
streamlit run app.py

# Run with minimal UI (headless mode)
uv run streamlit run app.py --client.toolbarMode=minimal --server.headless true --browser.gatherUsageStats false
```

### Dependency Management
```bash
# Install dependencies with uv
uv sync

# Install dev dependencies
uv sync --dev

# Install with pip
pip install -r requirements.txt
```

### Code Quality
```bash
# Run linter (ruff)
uv run ruff check .

# Format code with ruff
uv run ruff format .

# Run tests (if present)
uv run pytest
```

## Architecture Overview

### Core Application Flow
The application (`app.py`) is a Streamlit-based web app that implements an iterative LLM reasoning system:

1. **Multi-Model Support**: Integrates with Ollama (local), vLLM servers, and OpenRouter API through a unified `OpenAIReasoningClient` class
2. **Thinking Loop Process**: Implements iterative self-reflection where LLMs improve responses through multiple rounds
3. **Reasoning Extraction**: Captures explicit reasoning from `<think>` tags or native reasoning fields (e.g., OpenAI o1 models)
4. **YOLO Mode**: Automatic convergence detection using cosine similarity between iterations

### Key Components

#### Main Application (`app.py`)
- **OpenAIReasoningClient**: Handles API calls to different LLM providers
- **Thinking Loop Logic**: Core iteration and reflection mechanism
- **Session State Management**: Tracks experiments, iterations, and results
- **Auto-save System**: JSON-based experiment persistence in `saves/`

#### Visualization System (`src/visualizations.py`)
- **Async Computation**: ThreadPoolExecutor for parallel visualization generation
- **Caching Layer**: Thread-safe caching to optimize performance
- **Multiple Visualization Types**:
  - Concept evolution graphs (network analysis)
  - Similarity heatmaps (convergence tracking)
  - Confidence tracking (certainty/uncertainty markers)
  - Topic flow Sankey diagrams
  - Complexity metrics (vocabulary diversity)

#### Export System (`src/pdf_export.py`)
- **PDF Generation**: ReportLab-based professional report creation
- **Smart Filenames**: AI-generated descriptive names based on question content
- **HTML Export**: Alternative web-based report format
- **Comprehensive Reports**: Includes executive summary, visualizations, and full appendix

#### Prompt System
- **Templates Directory**: Pre-configured epistemic approaches (Socratic, empirical-scientific, dialectical, etc.)
- **Dynamic Loading**: Runtime template switching via UI
- **User Customization**: Editable prompts stored in `src/prompts.json`

### Data Flow
1. User inputs question → System prompt + reflection template applied
2. LLM generates initial response → Reasoning extracted (if available)
3. Reflection loop begins → Each iteration reviews previous response
4. Convergence detection (YOLO mode) or fixed iterations
5. Results saved to JSON → Visualizations computed asynchronously
6. Export to PDF/HTML with AI-generated filename

### Key Design Patterns
- **Separation of Concerns**: Visualization, export, and core logic in separate modules
- **Async Processing**: Non-blocking visualization computation
- **Template Pattern**: Swappable prompt templates for different reasoning approaches
- **Observer Pattern**: Real-time UI updates during iteration progress