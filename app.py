import json
import os
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
import streamlit as st
from openai import OpenAI
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import visualization module
from src.visualizations import (
    show_visualization_buttons,
    show_visualization_modals,
    clear_viz_cache,
    compute_all_visualizations_async
)

# Page configuration
st.set_page_config(
    page_title="LLM Reflection Lab",
    page_icon="🧠",
    layout="wide"
)

class OpenAIReasoningClient:
    """Client for interacting with OpenAI API with reasoning models"""
    
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
    
    def call_llm_with_reasoning(self, messages: List[Dict]) -> Dict:
        """Make a call to OpenAI API and extract reasoning content"""
        try:
            # Build parameters based on model type
            params = {
                "model": self.model,
                "messages": messages
            }
            
            # o1 models don't support temperature parameter
            if not (self.model.startswith("o1") or self.model.startswith("o3")):
                params["temperature"] = 0.7
            
            # For models like o1 that support reasoning
            response = self.client.chat.completions.create(**params)
            # print(response)
            # Extract the response
            choice = response.choices[0]
            message = choice.message
            content = message.content
            
            result = {
                "content": content,
                "reasoning": None,
                "usage": response.usage.model_dump() if response.usage else None
            }
            
            # Method 1: Check for native reasoning field (o1 models)
            # Check if choice has reasoning directly
            if hasattr(choice, 'reasoning') and choice.reasoning:
                result["reasoning"] = choice.reasoning
                result["response"] = content  # The main content is the response
            # Check if message has reasoning
            elif hasattr(message, 'reasoning') and message.reasoning:
                result["reasoning"] = message.reasoning
                result["response"] = content  # The main content is the response
            # Check if there's a separate reasoning field in the response
            elif hasattr(response, 'reasoning') and response.reasoning:
                result["reasoning"] = response.reasoning
                result["response"] = content  # The main content is the response
            
            # Method 2: Check if content starts with <thinking> tag
            elif content and "<think>" in content and "</think>" in content:
                try:
                    start = content.find("<think>") + len("<think>")
                    end = content.find("</think>")
                    reasoning = content[start:end].strip()
                    # Remove thinking tags from main content to get the response
                    response_content = content[:content.find("<think>")] + content[content.find("</think>") + len("</think>"):]
                    response_content = response_content.strip()
                    result["reasoning"] = reasoning
                    result["response"] = response_content
                except Exception:
                    # If extraction fails, leave reasoning as None
                    result["response"] = content
            else:
                # No reasoning found, content is the response
                result["response"] = content
            
            # Debug logging to see what fields are available
            if st.session_state.get("debug_mode", False):
                st.sidebar.write("Debug - Available fields in choice:", dir(choice))
                st.sidebar.write("Debug - Available fields in message:", dir(message))
                if result["reasoning"]:
                    st.sidebar.success(f"Reasoning extracted: {len(result['reasoning'])} chars")
                else:
                    st.sidebar.info("No reasoning content found")
            
            return result
            
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return None
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Return list of recommended models for reasoning"""
        return [
            "o1",
            "o1-mini",
            "o1-preview",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4",
            "gpt-3.5-turbo"
        ]

class ThinkingLoop:
    """Manages the thinking loop process"""
    
    def __init__(self, openai_client: OpenAIReasoningClient):
        self.client = openai_client
        self.iterations = []
    
    
    def create_reflection_prompt(self, reasoning: str, response: str, original_question: str, iteration: int) -> str:
        """Create a prompt asking the model to reflect on its thinking"""
        prompts = st.session_state.prompts
        if reasoning:
            # We successfully extracted reasoning, so we can reference it
            template = prompts["reflection_template"]["with_reasoning"]
            return template.format(
                original_question=original_question,
                reasoning=reasoning,
                response=response
            )
        else:
            # No reasoning was extracted, just work with the response
            template = prompts["reflection_template"]["without_reasoning"]
            return template.format(
                original_question=original_question,
                response=response
            )
    
    def run_iteration(self, question: str, previous_reasoning: str = "", previous_response: str = "", iteration_num: int = 1) -> Dict:
        """Run a single iteration of the thinking loop"""
        prompts = st.session_state.prompts
        if not previous_reasoning and not previous_response:
            # First iteration - just ask the question naturally
            messages = [{
                "role": "system",
                "content": prompts["system_prompts"]["initial"]
            }, {
                "role": "user",
                "content": question
            }]
        else:
            # Subsequent iterations - ask for reflection
            reflection_prompt = self.create_reflection_prompt(previous_reasoning, previous_response, question, iteration_num)
            messages = [{
                "role": "system",
                "content": prompts["system_prompts"]["reflection"]
            }, {
                "role": "user",
                "content": reflection_prompt
            }]
        
        # Track time for tokens/second calculation
        start_time = time.time()
        response = self.client.call_llm_with_reasoning(messages)
        elapsed_time = time.time() - start_time
        
        if response:
            iteration_data = {
                "timestamp": datetime.now().isoformat(),
                "reasoning": response.get("reasoning", ""),
                "response": response.get("response", response.get("content", "")),
                "full_response": response.get("content", ""),
                "iteration_number": iteration_num,
                "usage": response.get("usage"),
                "elapsed_time": elapsed_time  # Add elapsed time for tokens/sec calculation
            }
            self.iterations.append(iteration_data)
            return iteration_data
        
        return None
    
    def run_loop(self, question: str, num_iterations: int, progress_callback=None, yolo_mode: bool = False, convergence_threshold: float = 0.99) -> List[Dict]:
        """Run the complete thinking loop for specified iterations
        
        Args:
            question: The question to answer
            num_iterations: Maximum number of iterations
            progress_callback: Optional callback for progress updates
            yolo_mode: If True, stop early when convergence is detected
            convergence_threshold: Similarity threshold for convergence (default 0.99)
        """
        self.iterations = []
        reasoning = ""
        response = ""
        converged = False
        
        for i in range(num_iterations):
            if progress_callback:
                progress_callback(i + 1, num_iterations)
            
            iteration = self.run_iteration(question, reasoning, response, i + 1)
            
            if iteration:
                reasoning = iteration["reasoning"]
                response = iteration["response"]
                
                # Check for convergence in YOLO mode
                if yolo_mode and i > 0:  # Need at least 2 iterations to compare
                    prev_text = f"{self.iterations[-1]['reasoning']} {self.iterations[-1]['response']}"
                    curr_text = f"{iteration['reasoning']} {iteration['response']}"
                    
                    similarity = calculate_similarity(prev_text, curr_text)
                    iteration['similarity_to_previous'] = similarity
                    
                    if similarity >= convergence_threshold:
                        converged = True
                        st.success(f"🎯 YOLO Mode: Convergence detected at iteration {i + 1} (similarity: {similarity:.3f})")
                        break
            else:
                st.error(f"Failed at iteration {i + 1}")
                break
            
            # Small delay to avoid rate limiting
            time.sleep(1)
        
        if yolo_mode and not converged and len(self.iterations) == num_iterations:
            st.info(f"🎯 YOLO Mode: Completed all {num_iterations} iterations without convergence")
        
        return self.iterations

def load_prompts():
    """Load prompts from configuration file"""
    prompts_file = Path("prompts.json")
    if prompts_file.exists():
        with open(prompts_file, "r") as f:
            return json.load(f)
    else:
        # Default prompts if file doesn't exist
        return {
            "system_prompts": {
                "initial": "You are a helpful assistant. Think carefully and provide a thorough answer.",
                "reflection": "You are a helpful assistant. Reflect critically on your previous response and provide an improved, more insightful answer."
            },
            "reflection_template": {
                "with_reasoning": "Looking at this question again: {original_question}\n\nIn my previous attempt, I thought through it this way:\n{reasoning}\n\nAnd concluded:\n{response}\n\nLet me reconsider this problem. What did I miss? What assumptions did I make? Are there other perspectives or deeper insights I should explore? \n\nPlease provide an improved, more thorough answer that builds on or corrects my previous thinking.",
                "without_reasoning": "Looking at this question again: {original_question}\n\nMy previous answer was:\n{response}\n\nLet me think about this more deeply. What aspects did I not fully consider? What additional insights or perspectives would improve this answer?\n\nPlease provide a more comprehensive and refined response."
            }
        }

def save_prompts(prompts):
    """Save prompts to configuration file"""
    with open("prompts.json", "w") as f:
        json.dump(prompts, f, indent=2)

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "thinking_loops" not in st.session_state:
        st.session_state.thinking_loops = []
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""
    if "total_tokens_used" not in st.session_state:
        st.session_state.total_tokens_used = 0
    if "total_elapsed_time" not in st.session_state:
        st.session_state.total_elapsed_time = 0
    if "expanded_iteration" not in st.session_state:
        st.session_state.expanded_iteration = None
    if "page_number" not in st.session_state:
        st.session_state.page_number = 0
    if "prompts" not in st.session_state:
        st.session_state.prompts = load_prompts()
    if "show_edit_prompts" not in st.session_state:
        st.session_state.show_edit_prompts = False

def show_edit_prompts_modal():
    """Show modal for editing prompts"""
    @st.dialog("✏️ Edit Prompts", width="large")
    def edit_prompts_dialog():
        st.markdown("### System Prompts")
        
        # Create a copy of prompts for editing
        edited_prompts = st.session_state.prompts.copy()
        
        # System prompts
        edited_prompts["system_prompts"]["initial"] = st.text_area(
            "Initial Iteration Prompt",
            value=st.session_state.prompts["system_prompts"]["initial"],
            height=100,
            help="System prompt for the first iteration"
        )
        
        edited_prompts["system_prompts"]["reflection"] = st.text_area(
            "Reflection Iteration Prompt",
            value=st.session_state.prompts["system_prompts"]["reflection"],
            height=100,
            help="System prompt for reflection iterations"
        )
        
        st.markdown("### Reflection Templates")
        st.caption("💡 Use {original_question}, {reasoning}, and {response} as placeholders")
        
        edited_prompts["reflection_template"]["with_reasoning"] = st.text_area(
            "Reflection Template (with reasoning)",
            value=st.session_state.prompts["reflection_template"]["with_reasoning"],
            height=200,
            help="Template when reasoning was extracted from previous iteration"
        )
        
        edited_prompts["reflection_template"]["without_reasoning"] = st.text_area(
            "Reflection Template (without reasoning)",
            value=st.session_state.prompts["reflection_template"]["without_reasoning"],
            height=200,
            help="Template when no reasoning was extracted"
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("💾 Save", type="primary", use_container_width=True):
                st.session_state.prompts = edited_prompts
                save_prompts(edited_prompts)
                st.success("✅ Prompts saved successfully!")
                time.sleep(1)
                st.session_state.show_edit_prompts = False
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset to Defaults", use_container_width=True):
                default_prompts = {
                    "system_prompts": {
                        "initial": "You are a helpful assistant. Think carefully and provide a thorough answer.",
                        "reflection": "You are a helpful assistant. Reflect critically on your previous response and provide an improved, more insightful answer."
                    },
                    "reflection_template": {
                        "with_reasoning": "Looking at this question again: {original_question}\n\nIn my previous attempt, I thought through it this way:\n{reasoning}\n\nAnd concluded:\n{response}\n\nLet me reconsider this problem. What did I miss? What assumptions did I make? Are there other perspectives or deeper insights I should explore? \n\nPlease provide an improved, more thorough answer that builds on or corrects my previous thinking.",
                        "without_reasoning": "Looking at this question again: {original_question}\n\nMy previous answer was:\n{response}\n\nLet me think about this more deeply. What aspects did I not fully consider? What additional insights or perspectives would improve this answer?\n\nPlease provide a more comprehensive and refined response."
                    },
                    "reasoning_extraction": {
                        "think_tag_start": "<think>",
                        "think_tag_end": "</think>",
                        "reasoning_field": "reasoning"
                    },
                    "ui_messages": {
                        "test_connection": "Say 'Connection successful' if you can hear me.",
                        "no_reasoning_found": "*No explicit reasoning captured*",
                        "no_response": "No response"
                    }
                }
                st.session_state.prompts = default_prompts
                save_prompts(default_prompts)
                st.success("🔄 Reset to default prompts")
                time.sleep(1)
                st.rerun()
        
        with col3:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.show_edit_prompts = False
                st.rerun()
    
    # Show the dialog
    if st.session_state.show_edit_prompts:
        edit_prompts_dialog()

def main():
    initialize_session_state()
    
    # Show edit prompts modal if requested
    show_edit_prompts_modal()
    
    st.title("🧠 LLM Reflection Lab")
    st.markdown("Watch as the LLM iteratively refines its thinking and responses through self-reflection")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Ollama/OpenAI settings
        st.subheader("LLM Settings")
        
        # Server type selection
        server_type = st.radio(
            "Server Type",
            ["Ollama (Local)", "vLLM", "OpenRouter"],
            help="Choose between local Ollama, vLLM server, or OpenRouter API"
        )
        
        if server_type == "Ollama (Local)":
            # Ollama configuration
            ollama_url = st.text_input(
                "Ollama URL",
                value="http://localhost:11434",
                help="Ollama server URL (without /v1)"
            )
            
            # Load models button
            if st.button("🔄 Load Available Models"):
                try:
                    # Fetch models from Ollama
                    response = requests.get(f"{ollama_url}/api/tags")
                    if response.status_code == 200:
                        models_data = response.json()
                        available_models = [model["name"] for model in models_data.get("models", [])]
                        if available_models:
                            st.session_state.ollama_models = available_models
                            st.success(f"✅ Found {len(available_models)} models")
                        else:
                            st.warning("No models found. Please pull a model first.")
                    else:
                        st.error(f"Failed to fetch models: {response.status_code}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
            
            # Model selection from fetched models
            if "ollama_models" in st.session_state and st.session_state.ollama_models:
                model_name = st.selectbox(
                    "Select Model",
                    options=st.session_state.ollama_models,
                    help="Select from available Ollama models"
                )
            else:
                model_name = st.text_input(
                    "Model Name",
                    value="gpt-oss-20b",
                    help="Enter model name manually or click 'Load Available Models'"
                )
            
            # Set OpenAI-compatible endpoint for Ollama
            base_url = f"{ollama_url}/v1"
            api_key = "ollama"  # Ollama doesn't require an API key, but OpenAI client needs something
            
        elif server_type == "vLLM":  # vLLM Server
            # vLLM configuration
            vllm_url = st.text_input(
                "vLLM Server URL",
                value="http://localhost:8000",
                help="vLLM server URL (without /v1)"
            )
            
            # API Key for vLLM
            vllm_api_key = st.text_input(
                "vLLM API Key",
                type="password",
                value=os.getenv("VLLM_API_KEY", ""),
                help="Enter your vLLM API key"
            )
            
            # Load models button
            if st.button("🔄 Load Available Models"):
                try:
                    # Try the OpenAI-compatible models endpoint
                    headers = {"Authorization": f"Bearer {vllm_api_key}"} if vllm_api_key else {}
                    response = requests.get(f"{vllm_url}/v1/models", headers=headers)
                    if response.status_code == 200:
                        models_data = response.json()
                        available_models = [model["id"] for model in models_data.get("data", [])]
                        if available_models:
                            st.session_state.vllm_models = available_models
                            st.success(f"✅ Found {len(available_models)} models")
                        else:
                            st.warning("No models found on vLLM server.")
                    else:
                        st.error(f"Failed to fetch models: {response.status_code}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
            
            # Model selection from fetched models
            if "vllm_models" in st.session_state and st.session_state.vllm_models:
                model_name = st.selectbox(
                    "Select Model",
                    options=st.session_state.vllm_models,
                    help="Select from available vLLM models"
                )
            else:
                model_name = st.text_input(
                    "Model Name",
                    value="meta-llama/Llama-2-7b-chat-hf",
                    help="Enter model name manually or click 'Load Available Models'"
                )
            
            # Set OpenAI-compatible endpoint for vLLM
            base_url = f"{vllm_url}/v1"
            api_key = vllm_api_key if vllm_api_key else "dummy"  # vLLM might require an API key
            
        elif server_type == "OpenRouter":  # OpenRouter API
            # OpenRouter configuration
            st.info("🌐 OpenRouter provides access to many models via a unified API")
            
            # API Key for OpenRouter
            openrouter_api_key = st.text_input(
                "OpenRouter API Key",
                type="password",
                value=os.getenv("OPENROUTER_API_KEY", ""),
                help="Enter your OpenRouter API key (get one at openrouter.ai)"
            )
            
            # Common OpenRouter models that support reasoning
            default_models = [
                "deepseek/deepseek-r1-0528:free",
                "openai/gpt-oss-20b:free",
                "qwen/qwen3-235b-a22b:free"
            ]
            
            # Model selection
            model_input_method = st.radio(
                "Model Selection",
                ["Choose from list", "Enter custom model"],
                horizontal=True
            )
            
            if model_input_method == "Choose from list":
                model_name = st.selectbox(
                    "Select Model",
                    options=default_models,
                    help="Select from common OpenRouter models"
                )
            else:
                model_name = st.text_input(
                    "Model Name",
                    value="deepseek/deepseek-r1-0107:free",
                    help="Enter the OpenRouter model name (e.g., provider/model-name)"
                )
            
            # Set OpenRouter endpoint
            base_url = "https://openrouter.ai/api/v1"
            api_key = openrouter_api_key
        
        # Loop settings
        st.subheader("Loop Settings")
        num_iterations = st.slider(
            "Number of Iterations", 
            min_value=1, 
            max_value=100, 
            value=3,
            help="Number of thinking iterations to refine the response"
        )
        
        # YOLO Mode toggle
        yolo_mode = st.checkbox(
            "🎯 YOLO Mode (Run until convergence)", 
            value=False,
            help="Run iterations indefinitely until consecutive iterations reach similarity threshold. Ignores iteration count."
        )
        
        convergence_threshold = 0.99  # Default value
        if yolo_mode:
            convergence_threshold = st.slider(
                "Convergence Threshold",
                min_value=0.80,
                max_value=0.99,
                value=0.99,
                step=0.01,
                help="Similarity threshold to detect convergence (0.99 = 99% similar)"
            )
            st.info(f"🎯 YOLO Mode: Will run indefinitely until convergence is detected (similarity ≥ {convergence_threshold:.0%}). Iteration slider is ignored. Safety limit: 100 iterations.")
        
        # Display token usage and performance metrics
        if st.session_state.total_tokens_used > 0:
            st.subheader("📊 Performance Metrics")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Total Tokens", f"{st.session_state.total_tokens_used:,}")
            with col_m2:
                # Calculate tokens per second if we have timing data
                if "total_elapsed_time" in st.session_state and st.session_state.total_elapsed_time > 0:
                    tokens_per_sec = st.session_state.total_tokens_used / st.session_state.total_elapsed_time
                    st.metric("Tokens/Second", f"{tokens_per_sec:.1f}")
                else:
                    st.metric("Tokens/Second", "N/A")
        
        # Debug mode toggle
        st.divider()
        debug_mode = st.checkbox("🐛 Debug Mode", value=st.session_state.get("debug_mode", False), 
                                help="Show debug information about API responses")
        st.session_state.debug_mode = debug_mode
        
        # Test connection
        if st.button("🔌 Test Connection"):
            if server_type == "OpenRouter" and not api_key:
                st.warning("Please enter your OpenRouter API key first.")
            elif server_type in ["Ollama (Local)", "vLLM", "OpenRouter"]:
                try:
                    client = OpenAIReasoningClient(api_key, model_name, base_url)
                    test_response = client.call_llm_with_reasoning([{"role": "user", "content": "Say 'Connection successful' if you can hear me."}])
                    if test_response:
                        st.success("✅ Connection successful!")
                        st.info(f"Response: {test_response.get('content', '')[:100]}...")
                    else:
                        st.error("❌ Connection failed.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("Please enter an API key first.")
    
    # Main content area
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        
        # Input area
        question = st.text_area(
            "Enter your question:",
            height=120,
            placeholder="Ask a complex question that would benefit from iterative thinking...\n\nExamples:\n- Explain quantum computing and its implications\n- Design a sustainable city of the future\n- Analyze the trolley problem from multiple ethical frameworks"
        )
        
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
        with col_btn1:
            start_button = st.button("🚀 Start Loop", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("✏️ Prompts", use_container_width=True):
                st.session_state.show_edit_prompts = True
                st.rerun()
        with col_btn3:
            clear_button = st.button("🧹 Clear", use_container_width=True)
        with col_btn4:
            export_button = st.button("📤 Export", use_container_width=True)
        
        if clear_button:
            # Clear all experiment data but keep configuration
            st.session_state.thinking_loops = []
            st.session_state.total_tokens_used = 0
            st.session_state.total_elapsed_time = 0
            st.session_state.page_number = 0
            st.session_state.expanded_iteration = None
            if "active_loop" in st.session_state:
                st.session_state.active_loop = None
            # Clear visualization cache
            clear_viz_cache()
            st.success("✅ Cleared all experiment data")
            time.sleep(0.5)
            st.rerun()
        
        if export_button:
            # Export the current experiment to PDF
            if st.session_state.thinking_loops or ("active_loop" in st.session_state and st.session_state.active_loop):
                export_to_pdf()
            else:
                st.warning("No experiments to export.")
        
        if start_button:
            if question and (server_type in ["Ollama (Local)", "vLLM"] or (server_type == "OpenRouter" and api_key)):
                st.session_state.current_question = question
                
                # Initialize a new thinking loop in session state
                if "active_loop" not in st.session_state:
                    st.session_state.active_loop = None
                
                st.session_state.active_loop = {
                    "question": question,
                    "iterations": [],
                    "timestamp": datetime.now().isoformat(),
                    "tokens_used": 0,
                    "is_running": True
                }
                st.rerun()
            elif server_type == "OpenRouter" and not api_key:
                st.warning("Please enter your OpenRouter API key.")
            else:
                st.warning("Please enter a question first.")
        
        # Handle active thinking loop
        if "active_loop" in st.session_state and st.session_state.active_loop and st.session_state.active_loop.get("is_running"):
            active_loop = st.session_state.active_loop
            
            # Add stop button for running loops
            if st.button("⏹️ Stop Current Loop", type="secondary", use_container_width=True):
                active_loop["is_running"] = False
                # Move to completed loops even if not all iterations done
                st.session_state.total_tokens_used += active_loop.get("tokens_used", 0)
                st.session_state.thinking_loops.append(active_loop)
                st.session_state.active_loop = None
                st.success(f"Stopped at iteration {len(active_loop['iterations'])}")
                time.sleep(1)
                st.rerun()
            
            # Create client and thinking loop
            client = OpenAIReasoningClient(api_key, model_name, base_url)
            thinking_loop = ThinkingLoop(client)
            
            # Progress indicator
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            # Run iterations one by one with display updates
            current_iteration = len(active_loop["iterations"])
            
            # Determine if we should continue
            should_continue = False
            if yolo_mode:
                # In YOLO mode, continue until convergence or safety limit
                should_continue = (current_iteration < 100) and (not active_loop.get("converged_early", False))
            else:
                # In normal mode, respect the iteration count
                should_continue = (current_iteration < num_iterations) and (not active_loop.get("converged_early", False))
            
            if should_continue:
                # Update progress
                if yolo_mode:
                    # In YOLO mode, show indeterminate progress
                    progress_bar.progress((current_iteration + 1) % 10 / 10)  # Cycling progress
                    status_text.text(f"🎯 YOLO Mode: Processing iteration {current_iteration + 1} (running until convergence)...")
                else:
                    progress_bar.progress((current_iteration + 1) / num_iterations)
                    status_text.text(f"🔄 Processing iteration {current_iteration + 1} of {num_iterations}...")
                
                # Get previous reasoning and response if available
                if active_loop["iterations"]:
                    prev_iter = active_loop["iterations"][-1]
                    prev_reasoning = prev_iter["reasoning"]
                    prev_response = prev_iter["response"]
                else:
                    prev_reasoning = ""
                    prev_response = ""
                
                # Run single iteration
                iteration = thinking_loop.run_iteration(
                    active_loop["question"],
                    prev_reasoning,
                    prev_response,
                    current_iteration + 1
                )
                
                if iteration:
                    # Check for convergence in YOLO mode
                    converged = False
                    if yolo_mode and current_iteration >= 1:  # Need at least 2 iterations to compare (0-indexed)
                        prev_text = f"{prev_reasoning} {prev_response}"
                        curr_text = f"{iteration['reasoning']} {iteration['response']}"
                        
                        similarity = calculate_similarity(prev_text, curr_text)
                        iteration['similarity_to_previous'] = similarity
                        
                        if similarity >= convergence_threshold:
                            converged = True
                            st.success(f"🎯 YOLO Mode: Convergence detected at iteration {current_iteration + 1} (similarity: {similarity:.3f} ≥ {convergence_threshold:.2f})")
                    
                    # Add iteration to active loop
                    active_loop["iterations"].append(iteration)
                    if iteration.get("usage"):
                        active_loop["tokens_used"] += iteration["usage"].get("total_tokens", 0)
                    if iteration.get("elapsed_time"):
                        active_loop["elapsed_time"] = active_loop.get("elapsed_time", 0) + iteration["elapsed_time"]
                    
                    # Debug: Show if reasoning was extracted
                    if st.session_state.get("debug_mode", False):
                        st.sidebar.info(f"Iteration {current_iteration + 1}: Reasoning extracted: {'Yes' if iteration['reasoning'] else 'No'}")
                        if not iteration['reasoning']:
                            st.sidebar.warning("No reasoning found - check response format")
                    
                    # Update session state
                    st.session_state.active_loop = active_loop
                    
                    # If converged in YOLO mode, mark as complete
                    if converged:
                        active_loop["is_running"] = False
                        active_loop["converged_early"] = True
                        active_loop["convergence_iteration"] = current_iteration + 1
                        active_loop["convergence_threshold"] = convergence_threshold
                        active_loop["final_similarity"] = similarity
                    
                    # Small delay for visibility
                    time.sleep(0.5)
                    
                    # Trigger rerun to show updates
                    st.rerun()
                else:
                    st.error(f"Failed at iteration {current_iteration + 1}")
                    active_loop["is_running"] = False
            else:
                # All iterations complete
                progress_bar.progress(1.0)
                status_text.text("")
                
                # Move to completed loops
                st.session_state.total_tokens_used += active_loop["tokens_used"]
                st.session_state.total_elapsed_time += active_loop.get("elapsed_time", 0)
                st.session_state.thinking_loops.append(active_loop)
                st.session_state.active_loop = None
                
                # Auto-save the experiment
                auto_save_experiment()
                
                # Pre-compute visualizations in background
                try:
                    compute_all_visualizations_async(active_loop['iterations'])
                except Exception:
                    pass  # Silent fail for background computation
                
                if active_loop.get("converged_early"):
                    final_sim = active_loop.get('final_similarity', 0.99)
                    threshold = active_loop.get('convergence_threshold', 0.99)
                    st.success(f"✅ YOLO Mode: Converged at iteration {active_loop['convergence_iteration']}! Final similarity: {final_sim:.3f} (threshold: {threshold:.2f}). Used {active_loop['tokens_used']:,} tokens.")
                elif yolo_mode and len(active_loop['iterations']) >= 100:
                    st.warning(f"⚠️ YOLO Mode: Reached safety limit of 100 iterations without convergence. Used {active_loop['tokens_used']:,} tokens.")
                else:
                    st.success(f"✅ Completed {len(active_loop['iterations'])} iterations! Used {active_loop['tokens_used']:,} tokens.")
                time.sleep(1)
                st.rerun()
        
        # Display latest thinking and response (including active loop)
        display_loop = None
        if "active_loop" in st.session_state and st.session_state.active_loop:
            display_loop = st.session_state.active_loop
            is_active = True
        elif st.session_state.thinking_loops:
            display_loop = st.session_state.thinking_loops[-1]
            is_active = False
        
        if display_loop and display_loop.get("iterations"):
            st.divider()
            if is_active:
                st.subheader("🔄 Current Progress")
            else:
                st.subheader("📝 Latest Result")
            
            # Show the question
            with st.container():
                st.markdown(f"**Question:** {display_loop['question']}")
                st.caption(f"{'Running' if is_active else 'Completed'} {len(display_loop['iterations'])} iterations • {display_loop.get('tokens_used', 0):,} tokens")
            
            # Show the latest iteration
            if display_loop['iterations']:
                latest_iteration = display_loop['iterations'][-1]
                
                # Display latest reasoning
                with st.expander(f"🧠 {'Current' if is_active else 'Final'} Reasoning (Iteration {latest_iteration['iteration_number']})", expanded=True):
                    if latest_iteration['reasoning']:
                        st.markdown(latest_iteration['reasoning'])
                    else:
                        st.info("*No explicit reasoning extracted from the model's response*")
                
                # Display latest response
                st.markdown(f"**{'Current' if is_active else 'Final'} Response:**")
                with st.container():
                    st.markdown(latest_iteration['response'] if latest_iteration['response'] else latest_iteration['full_response'])
    
    with col2:
        st.header("📈 Evolution of Thinking")
        
        # Determine which loop to display (active or latest completed)
        evolution_loop = None
        if "active_loop" in st.session_state and st.session_state.active_loop:
            evolution_loop = st.session_state.active_loop
            is_evolving = True
        elif st.session_state.thinking_loops:
            evolution_loop = st.session_state.thinking_loops[-1]
            is_evolving = False
        
        # Show visualization buttons and modals if we have iterations
        if evolution_loop and evolution_loop.get('iterations'):
            show_visualization_buttons(evolution_loop['iterations'])
            show_visualization_modals(evolution_loop['iterations'])
        
        if evolution_loop and evolution_loop.get('iterations'):
            if is_evolving:
                st.caption(f"🔄 Live updates - Iteration {len(evolution_loop['iterations'])} of {num_iterations}")
            else:
                st.caption(f"✅ Completed {len(evolution_loop['iterations'])} iterations")
            
            # Pagination controls for large iteration counts
            items_per_page = 10
            total_iterations = len(evolution_loop['iterations'])
            total_pages = (total_iterations - 1) // items_per_page + 1
            
            # Initialize page number if needed
            if "page_number" not in st.session_state:
                st.session_state.page_number = 0
            
            # Ensure page number is valid
            if st.session_state.page_number >= total_pages:
                st.session_state.page_number = total_pages - 1
            if st.session_state.page_number < 0:
                st.session_state.page_number = 0
            
            if total_iterations > items_per_page:
                col_page1, col_page2, col_page3 = st.columns([1, 2, 1])
                with col_page1:
                    if st.button("◀ Previous", disabled=st.session_state.page_number == 0, use_container_width=True):
                        st.session_state.page_number -= 1
                        st.rerun()
                with col_page2:
                    st.caption(f"Page {st.session_state.page_number + 1} of {total_pages} ({total_iterations} iterations)")
                with col_page3:
                    if st.button("Next ▶", disabled=st.session_state.page_number >= total_pages - 1, use_container_width=True):
                        st.session_state.page_number += 1
                        st.rerun()
                
                # Calculate page range
                start_idx = st.session_state.page_number * items_per_page
                end_idx = min(start_idx + items_per_page, total_iterations)
                iterations_to_show = evolution_loop['iterations'][start_idx:end_idx]
            else:
                iterations_to_show = evolution_loop['iterations']
                st.session_state.page_number = 0
            
            # Create container for iterations
            iterations_container = st.container()
            
            with iterations_container:
                # Display paginated iterations
                for idx, iteration in enumerate(iterations_to_show):
                    # Calculate actual iteration index
                    actual_idx = start_idx + idx if total_iterations > items_per_page else idx
                    # Accordion behavior - maximum one expanded at a time (can have all collapsed)
                    iteration_key = f"iter_{actual_idx}"
                    
                    # Start with all iterations collapsed (no auto-expand)
                    
                    # Create expander for each iteration with click handler
                    similarity_text = ""
                    if iteration.get('similarity_to_previous') is not None:
                        similarity_text = f" | Similarity: {iteration['similarity_to_previous']:.3f}"
                    
                    if st.button(
                        f"🔄 Iteration {iteration['iteration_number']} "
                        f"{'(Latest)' if actual_idx == len(evolution_loop['iterations']) - 1 else ''} "
                        f"- {iteration.get('usage', {}).get('total_tokens', 0):,} tokens{similarity_text}",
                        key=f"btn_{iteration_key}",
                        use_container_width=True
                    ):
                        # Toggle expansion state: clicking same = collapse, clicking different = expand new (auto-collapse others)
                        if st.session_state.expanded_iteration == iteration_key:
                            # Clicking the currently expanded item - collapse it (allow all collapsed)
                            st.session_state.expanded_iteration = None
                        else:
                            # Clicking a different item - expand it and collapse any others
                            st.session_state.expanded_iteration = iteration_key
                        st.rerun()
                    
                    # Show content if expanded
                    if st.session_state.expanded_iteration == iteration_key:
                        with st.container():
                            # Timestamp
                            st.caption(f"⏰ {iteration['timestamp']}")
                            
                            # Show reasoning
                            st.markdown("### 🧠 Reasoning Process")
                            if iteration['reasoning']:
                                with st.container():
                                    # Display reasoning with proper markdown formatting
                                    reasoning_container = st.container()
                                    with reasoning_container:
                                        # Use markdown for proper formatting
                                        st.markdown(iteration['reasoning'])
                            else:
                                st.info("*No explicit reasoning captured*")
                            
                            # Show response
                            st.markdown("### 💡 Response")
                            response_text = iteration['response'] if iteration['response'] else iteration['full_response']
                            if response_text:
                                with st.container():
                                    # Display response with proper markdown formatting
                                    response_container = st.container()
                                    with response_container:
                                        # Use markdown for proper formatting
                                        st.markdown(response_text)
                            
                            # Show evolution metrics if not the first iteration
                            if actual_idx > 0:
                                st.markdown("### 📊 Evolution from Previous")
                                prev_iteration = evolution_loop['iterations'][actual_idx-1]
                            
                                # Calculate metrics
                                curr_reasoning_len = len(iteration['reasoning']) if iteration['reasoning'] else 0
                                prev_reasoning_len = len(prev_iteration['reasoning']) if prev_iteration['reasoning'] else 0
                                
                                curr_resp_len = len(iteration['response'] if iteration['response'] else iteration['full_response'])
                                prev_resp_len = len(prev_iteration['response'] if prev_iteration['response'] else prev_iteration['full_response'])
                                
                                # Display metrics
                                metric_col1, metric_col2, metric_col3 = st.columns(3)
                                with metric_col1:
                                    reasoning_delta = curr_reasoning_len - prev_reasoning_len
                                    st.metric(
                                        "Reasoning",
                                        f"{curr_reasoning_len:,} chars",
                                        f"{reasoning_delta:+,}"
                                    )
                                with metric_col2:
                                    response_delta = curr_resp_len - prev_resp_len
                                    st.metric(
                                        "Response",
                                        f"{curr_resp_len:,} chars",
                                        f"{response_delta:+,}"
                                    )
                                with metric_col3:
                                    if iteration.get('usage') and prev_iteration.get('usage'):
                                        curr_tokens = iteration['usage'].get('total_tokens', 0)
                                        prev_tokens = prev_iteration['usage'].get('total_tokens', 0)
                                        token_delta = curr_tokens - prev_tokens
                                        st.metric(
                                            "Tokens Used",
                                            f"{curr_tokens:,}",
                                            f"{token_delta:+,}"
                                        )
        else:
            st.info("👋 No thinking loops yet. Start by asking a question!")
            st.markdown("""
            ### How it works:
            1. **Ask a question** - Preferably something complex that benefits from deep thinking
            2. **Watch iterations** - The model will reflect on its own reasoning
            3. **See evolution** - Track how thoughts and responses improve
            
            ### Best for:
            - Complex problem solving
            - Philosophical questions
            - Creative tasks
            - Analysis and reasoning
            """)
    
    # History section at the bottom
    if st.session_state.thinking_loops and len(st.session_state.thinking_loops) > 1:
        st.divider()
        st.header("📚 Previous Questions")
        
        for loop in reversed(st.session_state.thinking_loops[:-1]):
            with st.expander(
                f"Q: {loop['question'][:80]}..." if len(loop['question']) > 80 else f"Q: {loop['question']}",
                expanded=False
            ):
                col_hist1, col_hist2, col_hist3 = st.columns(3)
                with col_hist1:
                    st.caption(f"⏰ {loop['timestamp'][:19]}")
                with col_hist2:
                    st.caption(f"🔄 {len(loop['iterations'])} iterations")
                with col_hist3:
                    st.caption(f"🪙 {loop.get('tokens_used', 0):,} tokens")
                
                if loop['iterations']:
                    final = loop['iterations'][-1]
                    st.markdown("**Final Response:**")
                    st.markdown(final['response'] if final['response'] else final['full_response'])

def auto_save_experiment():
    """Auto-save the current experiment to a JSON file in the saves folder"""
    try:
        # Create saves directory if it doesn't exist
        saves_dir = Path("saves")
        saves_dir.mkdir(exist_ok=True)
        
        # Prepare data to save
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "total_tokens_used": st.session_state.get("total_tokens_used", 0),
            "total_elapsed_time": st.session_state.get("total_elapsed_time", 0),
            "thinking_loops": st.session_state.get("thinking_loops", []),
        }
        
        # Create filename with timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = saves_dir / f"auto_save_{timestamp_str}.json"
        
        # Save to JSON file
        with open(filename, "w") as f:
            json.dump(save_data, f, indent=2)
        
        return filename
    except Exception:
        # Silent fail for auto-save
        return None

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using TF-IDF and cosine similarity"""
    if not text1 or not text2:
        return 0.0
    
    try:
        # Use TF-IDF for text vectorization
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return float(similarity_matrix[0][1])
    except:
        # Fallback for edge cases
        return 1.0 if text1 == text2 else 0.0

def markdown_to_html(text):
    """Convert markdown text to HTML"""
    if not text:
        return ""
    
    # Escape HTML special characters first
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Convert code blocks
    text = re.sub(r'```(\w+)?\n(.*?)\n```', r'<pre><code class="\1">\2</code></pre>', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    
    # Convert headers
    text = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    
    # Convert bold and italic
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<strong><em>\1</em></strong>', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    text = re.sub(r'__(.*?)__', r'<strong>\1</strong>', text)
    text = re.sub(r'_(.*?)_', r'<em>\1</em>', text)
    
    # Convert lists
    lines = text.split('\n')
    in_list = False
    result = []
    
    for line in lines:
        # Unordered lists
        if re.match(r'^\s*[-*+]\s+', line):
            if not in_list:
                result.append('<ul>')
                in_list = 'ul'
            item = re.sub(r'^\s*[-*+]\s+', '', line)
            result.append(f'<li>{item}</li>')
        # Ordered lists
        elif re.match(r'^\s*\d+\.\s+', line):
            if not in_list:
                result.append('<ol>')
                in_list = 'ol'
            item = re.sub(r'^\s*\d+\.\s+', '', line)
            result.append(f'<li>{item}</li>')
        else:
            if in_list:
                result.append(f'</{in_list}>')
                in_list = False
            result.append(line)
    
    if in_list:
        result.append(f'</{in_list}>')
    
    text = '\n'.join(result)
    
    # Convert line breaks to paragraphs
    paragraphs = text.split('\n\n')
    text = ''.join([f'<p>{p}</p>' if not p.startswith('<') else p for p in paragraphs if p.strip()])
    
    return text

def export_to_pdf():
    """Export the current experiment to a readable HTML format"""
    try:
        loops = st.session_state.get("thinking_loops", [])
        if "active_loop" in st.session_state and st.session_state.active_loop:
            loops = loops + [st.session_state.active_loop]
        
        if not loops:
            st.warning("No experiments to export.")
            return
        
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>LLM Reflection Lab Report</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            line-height: 1.6; 
            margin: 40px; 
            color: #333;
        }
        h1 { 
            color: #2c3e50; 
            border-bottom: 3px solid #3498db; 
            padding-bottom: 10px; 
        }
        h2 { 
            color: #34495e; 
            margin-top: 30px; 
            background: #ecf0f1;
            padding: 10px;
            border-radius: 5px;
        }
        h3 { 
            color: #7f8c8d; 
            margin-top: 20px;
        }
        .experiment { 
            border: 1px solid #bdc3c7; 
            padding: 20px; 
            margin: 20px 0; 
            border-radius: 8px;
            background: #fff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .iteration { 
            background: #f8f9fa; 
            padding: 15px; 
            margin: 15px 0; 
            border-left: 4px solid #3498db; 
            border-radius: 4px;
        }
        .reasoning { 
            background: #e8f6f3; 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 5px;
            border: 1px solid #a6d5cf;
        }
        .response { 
            background: #fef9e7; 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 5px;
            border: 1px solid #f7dc6f;
        }
        .metrics { 
            display: flex; 
            gap: 20px; 
            margin: 15px 0; 
        }
        .metric { 
            background: #ecf0f1; 
            padding: 10px 15px; 
            border-radius: 5px;
            font-weight: 600;
        }
        pre { 
            white-space: pre-wrap; 
            word-wrap: break-word; 
            font-family: 'Courier New', monospace;
            margin: 10px 0;
            background: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        code {
            background: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        ul, ol {
            margin: 10px 0;
            padding-left: 30px;
        }
        p {
            margin: 10px 0;
        }
        .summary {
            background: #d5dbdb;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <h1>🧠 LLM Reflection Lab Report</h1>
    <div class="summary">
        <p><strong>Generated:</strong> """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        <p><strong>Total Experiments:</strong> """ + str(len(loops)) + """</p>
        <p><strong>Total Tokens Used:</strong> """ + f"{st.session_state.get('total_tokens_used', 0):,}" + """</p>
        <p><strong>Total Time:</strong> """ + f"{st.session_state.get('total_elapsed_time', 0):.2f} seconds" + """</p>
    </div>
    """
        
        for exp_num, loop in enumerate(loops, 1):
            html += f"""
    <div class="experiment">
        <h2>Experiment {exp_num}</h2>
        <p><strong>Question:</strong> {loop.get('question', 'N/A')}</p>
        <p><strong>Timestamp:</strong> {loop.get('timestamp', 'N/A')}</p>
        <p><strong>Total Iterations:</strong> {len(loop.get('iterations', []))}</p>
        <p><strong>Tokens Used:</strong> {loop.get('tokens_used', 0):,}</p>
        """
            
            for iteration in loop.get('iterations', []):
                reasoning_html = markdown_to_html(iteration.get('reasoning', 'No reasoning captured'))
                response_html = markdown_to_html(iteration.get('response', iteration.get('full_response', 'No response')))
                
                html += f"""
        <div class="iteration">
            <h3>Iteration {iteration.get('iteration_number', 'N/A')}</h3>
            <div class="reasoning">
                <strong>🧠 Reasoning:</strong>
                <div>{reasoning_html}</div>
            </div>
            <div class="response">
                <strong>💡 Response:</strong>
                <div>{response_html}</div>
            </div>
            <div class="metrics">
                <span class="metric">Tokens: {iteration.get('usage', {}).get('total_tokens', 0):,}</span>
                """
                if iteration.get('elapsed_time'):
                    html += f"""<span class="metric">Time: {iteration['elapsed_time']:.2f}s</span>"""
                html += """
            </div>
        </div>
                """
            
            html += """</div>"""
        
        html += """
</body>
</html>"""
        
        # Create filename with timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_reflection_lab_report_{timestamp_str}.html"
        
        # Offer download
        st.download_button(
            label="📥 Download Report",
            data=html,
            file_name=filename,
            mime="text/html",
            use_container_width=True
        )
        
        st.success("✅ Report generated! Click above to download.")
        
    except Exception as e:
        st.error(f"Failed to export: {str(e)}")

if __name__ == "__main__":
    main()