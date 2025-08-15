"""
Visualization module for Thinking Loop Experiment
Provides various visualizations to analyze reasoning evolution
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re
import json
from typing import List, Dict, Any, Tuple
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor
import threading

# Cache for computed visualizations
_viz_cache = {}
_cache_lock = threading.Lock()

def clear_viz_cache():
    """Clear the visualization cache"""
    global _viz_cache
    with _cache_lock:
        _viz_cache.clear()

def get_cached_or_compute(key: str, compute_func, *args, **kwargs):
    """Get cached result or compute if not available"""
    with _cache_lock:
        if key not in _viz_cache:
            _viz_cache[key] = compute_func(*args, **kwargs)
        return _viz_cache[key]

# ===== CONCEPT EVOLUTION GRAPH =====

def extract_concepts(text: str, top_n: int = 10) -> List[str]:
    """Extract key concepts from text using TF-IDF"""
    # Simple word extraction (can be enhanced with spaCy later)
    words = re.findall(r'\b[a-z]+\b', text.lower())
    # Filter common words
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
                 'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
                 'me', 'my', 'your', 'our', 'their', 'its', 'let', 'if', 'then', 'else'}
    
    filtered_words = [w for w in words if w not in stopwords and len(w) > 3]
    word_counts = Counter(filtered_words)
    return [word for word, _ in word_counts.most_common(top_n)]

def build_concept_graph(iterations: List[Dict]) -> Tuple[nx.Graph, Dict]:
    """Build a concept evolution graph from iterations"""
    G = nx.Graph()
    concept_iterations = {}  # Track which iteration each concept first appeared
    
    for idx, iteration in enumerate(iterations):
        text = f"{iteration.get('reasoning', '')} {iteration.get('response', '')}"
        concepts = extract_concepts(text)
        
        # Add nodes
        for concept in concepts:
            if concept not in G:
                G.add_node(concept, first_iteration=idx+1, frequency=1)
                concept_iterations[concept] = idx+1
            else:
                G.nodes[concept]['frequency'] += 1
        
        # Add edges between concepts in same iteration
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i+1:]:
                if G.has_edge(c1, c2):
                    G[c1][c2]['weight'] += 1
                else:
                    G.add_edge(c1, c2, weight=1)
    
    return G, concept_iterations

def create_concept_evolution_graph(iterations: List[Dict]) -> str:
    """Create an interactive concept evolution graph"""
    G, concept_iterations = build_concept_graph(iterations)
    
    # Create Pyvis network
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    
    # Customize appearance
    for node in net.nodes:
        node_data = G.nodes[node['id']]
        # Color by iteration
        iteration = node_data['first_iteration']
        color_map = px.colors.sequential.Viridis
        color_idx = min(int((iteration - 1) / len(iterations) * len(color_map)), len(color_map) - 1)
        node['color'] = color_map[color_idx]
        # Size by frequency
        node['size'] = 10 + node_data['frequency'] * 3
        node['title'] = f"{node['id']}\nFirst: Iteration {iteration}\nFreq: {node_data['frequency']}"
    
    # Set physics options for better layout
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100},
            "barnesHut": {
                "gravitationalConstant": -8000,
                "springConstant": 0.04,
                "springLength": 100
            }
        }
    }
    """)
    
    # Generate HTML
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as f:
        net.save_graph(f.name)
        with open(f.name, 'r') as html_file:
            html_content = html_file.read()
        os.unlink(f.name)
    
    return html_content

# ===== SIMILARITY HEATMAP =====

def compute_similarity_matrix(iterations: List[Dict]) -> np.ndarray:
    """Compute pairwise similarity between iterations"""
    texts = []
    for iteration in iterations:
        text = f"{iteration.get('reasoning', '')} {iteration.get('response', '')}"
        texts.append(text)
    
    # Use TF-IDF for text vectorization
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
    except:
        # Fallback for very short texts
        similarity_matrix = np.eye(len(texts))
    
    return similarity_matrix

def create_similarity_heatmap(iterations: List[Dict]) -> go.Figure:
    """Create an interactive similarity heatmap"""
    similarity_matrix = compute_similarity_matrix(iterations)
    
    # Create labels
    labels = [f"Iter {i+1}" for i in range(len(iterations))]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=labels,
        y=labels,
        colorscale='Viridis',
        text=np.round(similarity_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Similarity"),
        hovertemplate='%{y} vs %{x}<br>Similarity: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Iteration Similarity Matrix",
        xaxis_title="Iteration",
        yaxis_title="Iteration",
        height=600,
        yaxis_autorange='reversed'
    )
    
    return fig

# ===== CONFIDENCE TRACKING =====

def extract_confidence_scores(text: str) -> Dict[str, float]:
    """Extract confidence indicators from text"""
    # Confidence markers
    high_confidence = ['certainly', 'definitely', 'clearly', 'obviously', 'undoubtedly', 
                       'absolutely', 'surely', 'without doubt', 'conclusively']
    medium_confidence = ['probably', 'likely', 'seems', 'appears', 'suggests', 
                        'indicates', 'generally', 'typically', 'usually']
    low_confidence = ['perhaps', 'maybe', 'possibly', 'might', 'could', 
                     'potentially', 'uncertain', 'unclear', 'ambiguous']
    
    text_lower = text.lower()
    word_count = len(text.split())
    
    # Count occurrences
    high_count = sum(1 for marker in high_confidence if marker in text_lower)
    medium_count = sum(1 for marker in medium_confidence if marker in text_lower)
    low_count = sum(1 for marker in low_confidence if marker in text_lower)
    
    # Calculate confidence score (0-100)
    if word_count == 0:
        return {'score': 50, 'high': 0, 'medium': 0, 'low': 0}
    
    confidence_score = 50  # Base score
    confidence_score += (high_count * 10) - (low_count * 10)
    confidence_score = max(0, min(100, confidence_score))
    
    return {
        'score': confidence_score,
        'high': high_count,
        'medium': medium_count,
        'low': low_count
    }

def create_confidence_tracking(iterations: List[Dict]) -> go.Figure:
    """Create confidence tracking visualization"""
    scores = []
    high_markers = []
    low_markers = []
    
    for iteration in iterations:
        text = f"{iteration.get('reasoning', '')} {iteration.get('response', '')}"
        confidence = extract_confidence_scores(text)
        scores.append(confidence['score'])
        high_markers.append(confidence['high'])
        low_markers.append(confidence['low'])
    
    x = list(range(1, len(iterations) + 1))
    
    # Create figure with subplots
    fig = go.Figure()
    
    # Main confidence score line
    fig.add_trace(go.Scatter(
        x=x, y=scores,
        mode='lines+markers',
        name='Confidence Score',
        line=dict(color='blue', width=3),
        marker=dict(size=10)
    ))
    
    # High confidence markers
    fig.add_trace(go.Bar(
        x=x, y=high_markers,
        name='High Confidence Words',
        marker_color='green',
        opacity=0.3,
        yaxis='y2'
    ))
    
    # Low confidence markers
    fig.add_trace(go.Bar(
        x=x, y=low_markers,
        name='Uncertainty Words',
        marker_color='red',
        opacity=0.3,
        yaxis='y2'
    ))
    
    # Add average line
    avg_score = np.mean(scores)
    fig.add_hline(y=avg_score, line_dash="dash", line_color="gray",
                  annotation_text=f"Average: {avg_score:.1f}")
    
    fig.update_layout(
        title="Confidence Evolution Across Iterations",
        xaxis_title="Iteration",
        yaxis_title="Confidence Score (0-100)",
        yaxis2=dict(
            title="Word Count",
            overlaying='y',
            side='right'
        ),
        height=500,
        hovermode='x unified'
    )
    
    return fig

# ===== TOPIC FLOW SANKEY DIAGRAM =====

def extract_topics(text: str, n_topics: int = 5) -> List[Tuple[str, float]]:
    """Extract main topics from text with weights"""
    # Simple topic extraction using word frequency
    # In production, use LDA or other topic modeling
    words = re.findall(r'\b[a-z]+\b', text.lower())
    
    # Filter stopwords and short words
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
                 'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
                 'me', 'my', 'your', 'our', 'their', 'its', 'let', 'if', 'then', 'else',
                 'more', 'most', 'less', 'very', 'much', 'many', 'some', 'any', 'all'}
    
    filtered_words = [w for w in words if w not in stopwords and len(w) > 3]
    
    if not filtered_words:
        return []
    
    # Count and normalize
    word_counts = Counter(filtered_words)
    total = sum(word_counts.values())
    
    # Get top topics with weights
    topics = [(word, count/total) for word, count in word_counts.most_common(n_topics)]
    return topics

def create_topic_flow_sankey(iterations: List[Dict]) -> go.Figure:
    """Create a Sankey diagram showing topic flow between iterations"""
    if len(iterations) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Need at least 2 iterations for topic flow",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    # Extract topics for each iteration
    iteration_topics = []
    for iteration in iterations:
        text = f"{iteration.get('reasoning', '')} {iteration.get('response', '')}"
        topics = extract_topics(text)
        iteration_topics.append(topics)
    
    # Build Sankey data
    sources = []
    targets = []
    values = []
    labels = []
    label_map = {}
    
    # Create nodes for each topic in each iteration
    node_counter = 0
    for iter_idx, topics in enumerate(iteration_topics):
        for topic, weight in topics:
            label = f"{topic} (Iter {iter_idx + 1})"
            if label not in label_map:
                label_map[label] = node_counter
                labels.append(label)
                node_counter += 1
    
    # Create links between consecutive iterations
    for i in range(len(iteration_topics) - 1):
        current_topics = dict(iteration_topics[i])
        next_topics = dict(iteration_topics[i + 1])
        
        # Link matching topics
        for topic in current_topics:
            if topic in next_topics:
                source_label = f"{topic} (Iter {i + 1})"
                target_label = f"{topic} (Iter {i + 2})"
                
                if source_label in label_map and target_label in label_map:
                    sources.append(label_map[source_label])
                    targets.append(label_map[target_label])
                    # Weight based on average presence
                    values.append((current_topics[topic] + next_topics[topic]) / 2)
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color='blue'
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color='rgba(0,0,255,0.2)'
        )
    )])
    
    fig.update_layout(
        title="Topic Flow Between Iterations",
        font_size=10,
        height=600
    )
    
    return fig

# ===== DIVERGENCE-CONVERGENCE TIMELINE =====

def calculate_divergence_score(iterations: List[Dict]) -> List[float]:
    """Calculate divergence score for each iteration"""
    if len(iterations) < 2:
        return [0.0] * len(iterations)
    
    scores = [0.0]  # First iteration has no divergence
    
    # Calculate semantic diversity for each iteration
    for i in range(1, len(iterations)):
        current_text = f"{iterations[i].get('reasoning', '')} {iterations[i].get('response', '')}"
        prev_text = f"{iterations[i-1].get('reasoning', '')} {iterations[i-1].get('response', '')}"
        
        # Use TF-IDF similarity as convergence measure
        try:
            vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([prev_text, current_text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Divergence is inverse of similarity
            divergence = 1 - similarity
            scores.append(divergence)
        except Exception:
            scores.append(0.5)  # Default middle value on error
    
    return scores

def create_divergence_convergence_timeline(iterations: List[Dict]) -> go.Figure:
    """Create timeline showing divergence and convergence patterns"""
    divergence_scores = calculate_divergence_score(iterations)
    x = list(range(1, len(iterations) + 1))
    
    # Calculate moving average for smoother trend
    window_size = min(3, len(divergence_scores))
    if len(divergence_scores) >= window_size:
        moving_avg = np.convolve(divergence_scores, 
                                 np.ones(window_size)/window_size, 
                                 mode='valid')
        # Pad to match original length
        moving_avg = np.pad(moving_avg, 
                           (window_size//2, window_size - window_size//2 - 1), 
                           mode='edge')
    else:
        moving_avg = divergence_scores
    
    # Create figure
    fig = go.Figure()
    
    # Add divergence line
    fig.add_trace(go.Scatter(
        x=x, y=divergence_scores,
        mode='lines+markers',
        name='Divergence Score',
        line=dict(color='red', width=2),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(255,0,0,0.1)'
    ))
    
    # Add trend line
    fig.add_trace(go.Scatter(
        x=x, y=moving_avg,
        mode='lines',
        name='Trend (Moving Avg)',
        line=dict(color='blue', width=3, dash='dash')
    ))
    
    # Add threshold lines
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray",
                  annotation_text="Neutral")
    fig.add_hline(y=0.3, line_dash="dot", line_color="green",
                  annotation_text="Convergence Zone")
    fig.add_hline(y=0.7, line_dash="dot", line_color="orange",
                  annotation_text="Divergence Zone")
    
    # Add annotations for phases
    # Identify exploration vs exploitation phases
    exploration_phases = []
    exploitation_phases = []
    current_phase = None
    phase_start = 0
    
    for i, score in enumerate(divergence_scores):
        if score > 0.6:  # Exploration
            if current_phase != 'exploration':
                if current_phase == 'exploitation':
                    exploitation_phases.append((phase_start, i-1))
                current_phase = 'exploration'
                phase_start = i
        elif score < 0.4:  # Exploitation
            if current_phase != 'exploitation':
                if current_phase == 'exploration':
                    exploration_phases.append((phase_start, i-1))
                current_phase = 'exploitation'
                phase_start = i
    
    # Close last phase
    if current_phase == 'exploration':
        exploration_phases.append((phase_start, len(divergence_scores)-1))
    elif current_phase == 'exploitation':
        exploitation_phases.append((phase_start, len(divergence_scores)-1))
    
    # Add phase annotations
    for start, end in exploration_phases:
        if start <= end:
            fig.add_vrect(x0=x[start]-0.5, x1=x[min(end, len(x)-1)]+0.5,
                         fillcolor="orange", opacity=0.1,
                         annotation_text="Exploring")
    
    for start, end in exploitation_phases:
        if start <= end:
            fig.add_vrect(x0=x[start]-0.5, x1=x[min(end, len(x)-1)]+0.5,
                         fillcolor="green", opacity=0.1,
                         annotation_text="Converging")
    
    fig.update_layout(
        title="Divergence-Convergence Timeline",
        xaxis_title="Iteration",
        yaxis_title="Divergence Score (0=Converged, 1=Diverged)",
        height=500,
        yaxis_range=[-0.1, 1.1],
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

# ===== COMPLEXITY METRICS =====

def calculate_complexity_metrics(text: str) -> Dict[str, float]:
    """Calculate text complexity metrics"""
    sentences = re.split(r'[.!?]+', text)
    words = text.split()
    
    # Basic metrics
    word_count = len(words)
    sentence_count = len([s for s in sentences if s.strip()])
    unique_words = len(set(words))
    
    # Avoid division by zero
    if sentence_count == 0:
        avg_sentence_length = 0
    else:
        avg_sentence_length = word_count / sentence_count
    
    if word_count == 0:
        vocabulary_diversity = 0
    else:
        vocabulary_diversity = unique_words / word_count
    
    # Count logical connectors
    logical_connectors = ['therefore', 'however', 'moreover', 'furthermore', 
                          'nevertheless', 'consequently', 'thus', 'hence',
                          'because', 'although', 'whereas', 'while']
    connector_count = sum(1 for word in words if word.lower() in logical_connectors)
    
    return {
        'word_count': word_count,
        'avg_sentence_length': avg_sentence_length,
        'vocabulary_diversity': vocabulary_diversity * 100,
        'logical_connectors': connector_count
    }

def create_complexity_metrics(iterations: List[Dict]) -> go.Figure:
    """Create complexity metrics visualization"""
    metrics_data = {
        'word_count': [],
        'avg_sentence_length': [],
        'vocabulary_diversity': [],
        'logical_connectors': []
    }
    
    for iteration in iterations:
        text = f"{iteration.get('reasoning', '')} {iteration.get('response', '')}"
        metrics = calculate_complexity_metrics(text)
        for key in metrics_data:
            metrics_data[key].append(metrics[key])
    
    x = list(range(1, len(iterations) + 1))
    
    # Create multi-line chart
    fig = go.Figure()
    
    # Normalize metrics for comparison
    for metric_name, values in metrics_data.items():
        if metric_name == 'word_count':
            continue  # Skip word count for clarity
        
        fig.add_trace(go.Scatter(
            x=x, y=values,
            mode='lines+markers',
            name=metric_name.replace('_', ' ').title(),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title="Reasoning Complexity Metrics",
        xaxis_title="Iteration",
        yaxis_title="Metric Value",
        height=500,
        hovermode='x unified'
    )
    
    return fig

# ===== MAIN VISUALIZATION INTERFACE =====

def show_visualization_buttons(iterations: List[Dict]):
    """Show visualization buttons above iteration display"""
    if not iterations:
        return
    
    st.markdown("### 📊 Visualizations")
    
    # First row of buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🕸️ Concept Graph", use_container_width=True):
            st.session_state.show_concept_graph = True
    
    with col2:
        if st.button("🔥 Similarity Map", use_container_width=True):
            st.session_state.show_similarity_map = True
    
    with col3:
        if st.button("📈 Confidence", use_container_width=True):
            st.session_state.show_confidence = True
    
    with col4:
        if st.button("📊 Complexity", use_container_width=True):
            st.session_state.show_complexity = True
    
    # Second row of buttons
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        if st.button("🌊 Topic Flow", use_container_width=True):
            st.session_state.show_topic_flow = True
    
    with col6:
        if st.button("↔️ Convergence", use_container_width=True):
            st.session_state.show_convergence = True

def show_visualization_modals(iterations: List[Dict]):
    """Display visualization modals"""
    
    # Concept Evolution Graph Modal
    if st.session_state.get('show_concept_graph', False):
        @st.dialog("🕸️ Concept Evolution Graph", width="large")
        def concept_graph_dialog():
            with st.spinner("Generating concept graph..."):
                try:
                    html_content = get_cached_or_compute(
                        f"concept_graph_{id(iterations)}",
                        create_concept_evolution_graph,
                        iterations
                    )
                    st.components.v1.html(html_content, height=650, scrolling=True)
                except Exception as e:
                    st.error(f"Error generating graph: {str(e)}")
            
            if st.button("Close", use_container_width=True):
                st.session_state.show_concept_graph = False
                st.rerun()
        
        concept_graph_dialog()
    
    # Similarity Heatmap Modal
    if st.session_state.get('show_similarity_map', False):
        @st.dialog("🔥 Iteration Similarity Heatmap", width="large")
        def similarity_map_dialog():
            with st.spinner("Computing similarity matrix..."):
                try:
                    fig = get_cached_or_compute(
                        f"similarity_{id(iterations)}",
                        create_similarity_heatmap,
                        iterations
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating heatmap: {str(e)}")
            
            if st.button("Close", use_container_width=True):
                st.session_state.show_similarity_map = False
                st.rerun()
        
        similarity_map_dialog()
    
    # Confidence Tracking Modal
    if st.session_state.get('show_confidence', False):
        @st.dialog("📈 Confidence Evolution", width="large")
        def confidence_dialog():
            with st.spinner("Analyzing confidence markers..."):
                try:
                    fig = get_cached_or_compute(
                        f"confidence_{id(iterations)}",
                        create_confidence_tracking,
                        iterations
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Confidence score based on linguistic markers (certainly, perhaps, etc.)")
                except Exception as e:
                    st.error(f"Error generating chart: {str(e)}")
            
            if st.button("Close", use_container_width=True):
                st.session_state.show_confidence = False
                st.rerun()
        
        confidence_dialog()
    
    # Complexity Metrics Modal
    if st.session_state.get('show_complexity', False):
        @st.dialog("📊 Complexity Metrics", width="large")
        def complexity_dialog():
            with st.spinner("Calculating complexity metrics..."):
                try:
                    fig = get_cached_or_compute(
                        f"complexity_{id(iterations)}",
                        create_complexity_metrics,
                        iterations
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Metrics: sentence length, vocabulary diversity, logical connectors")
                except Exception as e:
                    st.error(f"Error generating chart: {str(e)}")
            
            if st.button("Close", use_container_width=True):
                st.session_state.show_complexity = False
                st.rerun()
        
        complexity_dialog()
    
    # Topic Flow Sankey Modal
    if st.session_state.get('show_topic_flow', False):
        @st.dialog("🌊 Topic Flow Sankey Diagram", width="large")
        def topic_flow_dialog():
            with st.spinner("Analyzing topic transitions..."):
                try:
                    fig = get_cached_or_compute(
                        f"topic_flow_{id(iterations)}",
                        create_topic_flow_sankey,
                        iterations
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Shows how topics persist or change between iterations")
                except Exception as e:
                    st.error(f"Error generating diagram: {str(e)}")
            
            if st.button("Close", use_container_width=True):
                st.session_state.show_topic_flow = False
                st.rerun()
        
        topic_flow_dialog()
    
    # Divergence-Convergence Timeline Modal
    if st.session_state.get('show_convergence', False):
        @st.dialog("↔️ Divergence-Convergence Timeline", width="large")
        def convergence_dialog():
            with st.spinner("Calculating divergence patterns..."):
                try:
                    fig = get_cached_or_compute(
                        f"convergence_{id(iterations)}",
                        create_divergence_convergence_timeline,
                        iterations
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Higher scores = exploring new ideas, Lower scores = converging on solution")
                except Exception as e:
                    st.error(f"Error generating timeline: {str(e)}")
            
            if st.button("Close", use_container_width=True):
                st.session_state.show_convergence = False
                st.rerun()
        
        convergence_dialog()

# ===== ASYNC COMPUTATION SUPPORT =====

def compute_all_visualizations_async(iterations: List[Dict]):
    """Pre-compute all visualizations in background"""
    with ThreadPoolExecutor(max_workers=6) as executor:
        # Submit all computation tasks
        futures = {
            'concept': executor.submit(create_concept_evolution_graph, iterations),
            'similarity': executor.submit(create_similarity_heatmap, iterations),
            'confidence': executor.submit(create_confidence_tracking, iterations),
            'complexity': executor.submit(create_complexity_metrics, iterations),
            'topic_flow': executor.submit(create_topic_flow_sankey, iterations),
            'convergence': executor.submit(create_divergence_convergence_timeline, iterations)
        }
        
        # Store results in cache as they complete
        for name, future in futures.items():
            try:
                result = future.result(timeout=10)
                with _cache_lock:
                    _viz_cache[f"{name}_{id(iterations)}"] = result
            except Exception as e:
                print(f"Error computing {name}: {e}")