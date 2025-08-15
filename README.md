# 🧠 LLM Reflection Lab

An interactive research tool for exploring recursive reasoning in Large Language Models through iterative self-reflection. Watch as LLMs refine their thinking across multiple iterations without external feedback.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.40%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 Overview

This project implements a **thinking loop** system where LLMs iteratively improve their responses through self-reflection. Unlike traditional single-shot prompting, this approach allows models to:

- 🔄 **Reflect** on their previous reasoning
- 🎯 **Identify** gaps and assumptions  
- 📈 **Refine** their answers progressively
- 🧩 **Explore** different reasoning paths
<img width="1652" height="843" alt="Screenshot 2025-08-15 at 10 49 42 AM" src="https://github.com/user-attachments/assets/2f62a3a7-845c-482b-84c8-89c49a041d25" />
## ✨ Features

### Core Functionality
- **Multi-Model Support**: Works with Ollama, vLLM, and OpenRouter APIs
- **Reasoning Extraction**: Captures explicit reasoning from `<think>` tags or native fields
- **🎯 YOLO Mode**: Run iterations until convergence is detected automatically
- **Customizable Prompts**: Edit system prompts and reflection templates via UI
- **Auto-Save**: Experiments saved automatically in JSON format
- **Export**: Generate HTML reports of complete experiments

### 📊 Interactive Visualizations
- **🕸️ Concept Evolution Graph**: Network showing how concepts emerge and connect
- **🔥 Similarity Heatmap**: Matrix of iteration similarities to identify convergence
- **📈 Confidence Tracking**: Evolution of certainty/uncertainty markers
- **📊 Complexity Metrics**: Vocabulary diversity and logical connector usage
- **🌊 Topic Flow Sankey**: How topics persist or change between iterations
- **↔️ Convergence Timeline**: Tracks exploration vs exploitation phases
<img width="740" height="742" alt="Screenshot 2025-08-15 at 10 51 21 AM" src="https://github.com/user-attachments/assets/0239ef23-83aa-430c-afe6-fcfb93e75124" />

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- One of: Ollama, vLLM server, or OpenRouter API key

### Installation

#### Using uv (Recommended)
```bash
# Clone the repository
git clone https://github.com/chheplo/llm-reflection-lab.git
cd llm-reflection-lab

# Install with uv
uv sync
```

#### Using pip
```bash
# Clone the repository
git clone https://github.com/chheplo/llm-reflection-lab.git
cd llm-reflection-lab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# With uv
uv run streamlit run app.py

# With pip
streamlit run app.py
```

#### Alternative: Run with minimal UI and headless mode
```bash
# With uv (minimal toolbar, headless server, no usage stats)
uv run streamlit run app.py --client.toolbarMode=minimal --server.headless true --browser.gatherUsageStats false

# With pip
streamlit run app.py --client.toolbarMode=minimal --server.headless true --browser.gatherUsageStats false
```

The app will open at `http://localhost:8501`

## 🎮 Usage

### 1. Configure Your Model

#### For Ollama (Local)
- Install [Ollama](https://ollama.ai)
- Pull a model: `ollama pull gpt-oss:20b`
- Start Ollama: `ollama serve`
- Select "Ollama (Local)" in the app

#### For vLLM
- Start your vLLM server
- Enter the server URL and API key
- Click "Load Available Models"

#### For OpenRouter
- Get an API key from [OpenRouter](https://openrouter.ai)
- Select "OpenRouter" and enter your key
- Choose from available models

### 2. Run an Experiment

1. **Enter a Question**: Complex questions work best
2. **Set Iterations**: Choose 3-10 iterations (or more!)
3. **Optional - Enable YOLO Mode**: 
   - Toggle "🎯 YOLO Mode" to run until convergence
   - Adjust convergence threshold (0.80-0.99)
   - Iterations continue until consecutive responses are similar enough
4. **Start Loop**: Click to begin the thinking process
5. **Watch Evolution**: See reasoning improve in real-time
6. **Explore Visualizations**: Click visualization buttons for insights

### 3. Customize Prompts

Click "✏️ Prompts" to edit:
- System prompts for initial/reflection iterations
- Reflection templates with placeholders
- Save or reset to defaults

## 📁 Project Structure

```
llm-reflection-lab/
├── app.py                 # Main Streamlit application
├── src/
│   └── visualizations.py  # Visualization modules
├── prompts.json          # Customizable prompt templates
├── saves/                # Auto-saved experiments
├── pyproject.toml        # Project dependencies (uv)
├── requirements.txt      # Project dependencies (pip)
└── README.md            # This file
```

## 🔬 How It Works

### The Thinking Loop Process

1. **Initial Response**: Model answers the question
2. **Self-Reflection**: Model reviews its previous answer
3. **Improvement**: Model provides refined response
4. **Repeat**: Process continues for N iterations (or until convergence in YOLO Mode)

### Reasoning Extraction

The system extracts reasoning through:
- Native `reasoning` fields (e.g., OpenAI o1 models)
- `<think>...</think>` tags in responses
- Configurable extraction patterns

### Convergence Patterns

Through visualizations, you can observe:
- **Convergence**: Ideas stabilizing (high similarity)
- **Divergence**: Exploring new concepts (low similarity)
- **Phase Transitions**: Shifts between exploration/exploitation

### YOLO Mode (You Only Loop Once... Until Convergence)

When enabled, YOLO Mode:
- **Automatic Stopping**: Detects when consecutive iterations reach similarity threshold
- **Dynamic Duration**: Runs as many iterations as needed (up to 100 for safety)
- **Real-time Monitoring**: Shows convergence progress chart during execution
- **Efficiency**: Stops early when the model's responses stabilize
- **Configurable Threshold**: Adjust sensitivity from 80% to 99% similarity

## 📊 Example Insights

From a typical 10-iteration experiment:
- Iterations 1-2: Initial exploration
- Iterations 3-6: First convergence cluster
- Iteration 7: Divergence/pivot point
- Iterations 8-10: Final convergence

## 🛠️ Configuration

### Environment Variables
- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `VLLM_API_KEY`: Your vLLM server API key

### Prompt Customization
Edit `prompts.json` or use the UI to modify:
- System prompts
- Reflection templates
- Reasoning extraction patterns

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/llm-reflection-lab.git
cd thinking-loop-experiment

# Install in development mode
uv sync --dev

# Create a feature branch
git checkout -b feature/your-feature
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io)
- Visualizations powered by [Plotly](https://plotly.com) and [PyVis](https://pyvis.readthedocs.io)
- LLM integration via [OpenAI Python SDK](https://github.com/openai/openai-python)

## 📚 Citation

If you use this tool in your research, please cite:
```bibtex
@software{llm_reflection_lab,
  title = {LLM Reflection Lab},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/chheplo/llm-reflection-lab}
}
```

## 🔗 Links

- [Report Issues](https://github.com/chheplo/llm-reflection-lab/issues)
- [Documentation](https://github.com/chheplo/llm-reflection-lab/wiki)
- [Discussions](https://github.com/chheplo/llm-reflection-lab/discussions)

---

Made with ❤️ for the AI research community
