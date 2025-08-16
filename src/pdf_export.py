"""PDF Export Module for LLM Reflection Lab"""

import io
import base64
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.io as pio
import markdown2
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image, KeepTogether
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics import renderPDF
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus.flowables import Flowable


class NumberedCanvas(canvas.Canvas):
    """Canvas for adding page numbers"""
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        """Add page numbers and metadata to each page"""
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        """Draw page numbers at the bottom of each page"""
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.grey)
        self.drawRightString(
            letter[0] - 0.75 * inch,
            0.75 * inch,
            f"Page {self._pageNumber} of {page_count}"
        )
        self.drawString(
            0.75 * inch,
            0.75 * inch,
            "LLM Reflection Lab Report"
        )


class ChartFlowable(Flowable):
    """Custom flowable for embedding Plotly charts as images"""
    def __init__(self, fig, width=6*inch, height=4*inch):
        Flowable.__init__(self)
        self.fig = fig
        self.width = width
        self.height = height
        self.img_data = None
        
    def wrap(self, availWidth, availHeight):
        return self.width, self.height
    
    def draw(self):
        """Convert Plotly figure to image and draw it"""
        try:
            # Convert Plotly figure to PNG bytes
            img_bytes = self.fig.to_image(format="png", width=800, height=600, scale=2)
            img_reader = ImageReader(io.BytesIO(img_bytes))
            
            # Draw the image
            self.canv.drawImage(img_reader, 0, 0, width=self.width, height=self.height)
        except Exception as e:
            # If chart generation fails, draw a placeholder
            self.canv.setFont("Helvetica", 10)
            self.canv.drawString(10, self.height/2, f"[Chart could not be generated: {str(e)}]")


def generate_filename_from_question(question: str, llm_client=None) -> str:
    """Generate a smart filename based on the question content using the configured LLM"""
    
    # Try to use the provided LLM client first
    if llm_client:
        try:
            prompt = f"""Generate a filename for a PDF report about this question: "{question}"
            Requirements:
            - Maximum 5 words
            - End with "report"
            - Use hyphens between words
            - Keep it descriptive but concise
            - Use lowercase
            
            Example outputs: "quantum-computing-analysis-report", "sustainable-city-design-report", "ethics-trolley-problem-report"
            
            Output only the filename without extension, nothing else."""
            
            messages = [{"role": "user", "content": prompt}]
            response = llm_client.call_llm_with_reasoning(messages)
            
            if response and response.get('content'):
                # Extract just the filename from the response
                filename = response.get('response', response.get('content', '')).strip().lower()
                # Clean up - remove any explanatory text and just get the filename
                filename = filename.split('\n')[0]  # Take first line
                filename = re.sub(r'[^a-z0-9\-]', '', filename)
                
                # Ensure it ends with report
                if filename and not filename.endswith('report'):
                    filename = filename + '-report'
                
                if filename and len(filename) > 5:  # Reasonable filename
                    return filename
        except Exception:
            pass
    
    # Fallback: Create filename from question keywords
    try:
        # Extract key words from question
        words = re.findall(r'\b[a-z]+\b', question.lower())
        # Remove common words
        stopwords = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'was', 'were', 
                    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                    'of', 'with', 'by', 'from', 'can', 'you', 'explain', 'describe', 'tell',
                    'me', 'please', 'could', 'would', 'should', 'will', 'shall'}
        
        keywords = [w for w in words if w not in stopwords and len(w) > 2][:4]
        
        if keywords:
            filename = '-'.join(keywords) + '-report'
        else:
            filename = 'llm-reflection-report'
        
        return filename
    except:
        return 'llm-reflection-report'


def clean_text_for_pdf(text: str) -> str:
    """Clean and prepare text for PDF rendering"""
    if not text:
        return ""
    
    # Replace problematic characters
    text = text.replace('\u2019', "'")  # Right single quotation mark
    text = text.replace('\u201c', '"')  # Left double quotation mark
    text = text.replace('\u201d', '"')  # Right double quotation mark
    text = text.replace('\u2014', '--')  # Em dash
    text = text.replace('\u2013', '-')   # En dash
    text = text.replace('\u2026', '...')  # Ellipsis
    text = text.replace('\u00a0', ' ')   # Non-breaking space
    
    # Remove or replace other non-ASCII characters
    text = ''.join(char if ord(char) < 128 else ' ' for char in text)
    
    return text


def format_markdown_for_pdf(text: str, styles) -> List[Paragraph]:
    """Convert markdown text to ReportLab Paragraph objects using markdown2"""
    if not text:
        return [Paragraph("No content", styles['Normal'])]
    
    # Clean the text first
    text = clean_text_for_pdf(text)
    
    # Convert markdown to HTML using markdown2
    html_text = markdown2.markdown(
        text,
        extras=[
            'fenced-code-blocks',
            'tables',
            'break-on-newline',
            'code-friendly',
            'cuddled-lists',
            'smarty-pants'
        ]
    )
    
    # Process the HTML for ReportLab compatibility
    html_text = process_html_for_reportlab(html_text)
    
    # Split by block-level elements and create paragraphs
    paragraphs = []
    
    # Split by common block elements
    blocks = re.split(r'(<p>.*?</p>|<h[1-6]>.*?</h[1-6]>|<pre>.*?</pre>|<blockquote>.*?</blockquote>|<ul>.*?</ul>|<ol>.*?</ol>)', html_text, flags=re.DOTALL)
    
    for block in blocks:
        if not block or block.strip() == '':
            continue
            
        block = block.strip()
        
        # Headers
        if block.startswith('<h1>'):
            content = re.sub(r'</?h1>', '', block)
            paragraphs.append(Paragraph(f'<b><font size="16">{content}</font></b>', styles['Heading1']))
        elif block.startswith('<h2>'):
            content = re.sub(r'</?h2>', '', block)
            paragraphs.append(Paragraph(f'<b><font size="14">{content}</font></b>', styles['Heading2']))
        elif block.startswith('<h3>'):
            content = re.sub(r'</?h3>', '', block)
            paragraphs.append(Paragraph(f'<b><font size="12">{content}</font></b>', styles.get('Heading3', styles['Heading2'])))
        
        # Code blocks
        elif block.startswith('<pre>'):
            content = re.sub(r'</?pre>', '', block)
            content = re.sub(r'</?code[^>]*>', '', content)
            # Escape HTML entities in code
            content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            paragraphs.append(Paragraph(
                f'<font face="Courier" size="9" color="#333333">{content}</font>',
                styles.get('CodeStyle', styles['Normal'])
            ))
        
        # Blockquotes
        elif block.startswith('<blockquote>'):
            content = re.sub(r'</?blockquote>', '', block)
            content = re.sub(r'</?p>', '', content)
            paragraphs.append(Paragraph(
                f'<i><font color="#666666">{content}</font></i>',
                styles['Normal']
            ))
        
        # Lists
        elif block.startswith('<ul>'):
            # Process unordered list
            items = re.findall(r'<li>(.*?)</li>', block, re.DOTALL)
            for item in items:
                item = re.sub(r'</?p>', '', item).strip()
                paragraphs.append(Paragraph(f"• {item}", styles.get('BulletStyle', styles['Normal'])))
        
        elif block.startswith('<ol>'):
            # Process ordered list
            items = re.findall(r'<li>(.*?)</li>', block, re.DOTALL)
            for i, item in enumerate(items, 1):
                item = re.sub(r'</?p>', '', item).strip()
                paragraphs.append(Paragraph(f"{i}. {item}", styles.get('BulletStyle', styles['Normal'])))
        
        # Regular paragraphs
        elif block.startswith('<p>'):
            content = re.sub(r'</?p>', '', block)
            if content.strip():
                paragraphs.append(Paragraph(content, styles['Normal']))
        
        # Any other text
        elif block.strip() and not block.startswith('<'):
            paragraphs.append(Paragraph(block, styles['Normal']))
    
    return paragraphs if paragraphs else [Paragraph("No content", styles['Normal'])]


def process_html_for_reportlab(html_text: str) -> str:
    """Process HTML from markdown2 to be compatible with ReportLab"""
    # Convert strong to bold
    html_text = html_text.replace('<strong>', '<b>').replace('</strong>', '</b>')
    
    # Convert em to italic
    html_text = html_text.replace('<em>', '<i>').replace('</em>', '</i>')
    
    # Handle inline code
    html_text = re.sub(r'<code>([^<]+)</code>', r'<font face="Courier" color="#666666">\1</font>', html_text)
    
    # Remove links but keep text
    html_text = re.sub(r'<a[^>]*>([^<]+)</a>', r'\1', html_text)
    
    # Handle line breaks
    html_text = html_text.replace('\n', '<br/>')
    
    return html_text




def create_visualization_charts(iterations: List[Dict], similarity_mode: str = "response_only") -> Dict:
    """Create all visualization charts for the PDF report"""
    charts = {}
    
    # Import visualization functions from src.visualizations
    try:
        from src.visualizations import (
            create_similarity_heatmap,
            create_confidence_tracking,
            create_topic_flow_sankey,
            create_divergence_convergence_timeline,
            create_complexity_metrics
        )
    except ImportError:
        # Fallback to basic charts if imports fail
        pass
    
    try:
        # 1. Token usage over time chart
        if iterations:
            token_counts = [iter.get('usage', {}).get('total_tokens', 0) for iter in iterations]
            iteration_nums = list(range(1, len(iterations) + 1))
            
            fig_tokens = go.Figure()
            fig_tokens.add_trace(go.Scatter(
                x=iteration_nums,
                y=token_counts,
                mode='lines+markers',
                name='Tokens Used',
                line=dict(color='#3498db', width=2),
                marker=dict(size=8)
            ))
            fig_tokens.update_layout(
                title="Token Usage Per Iteration",
                xaxis_title="Iteration",
                yaxis_title="Tokens",
                height=400,
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            charts['token_usage'] = fig_tokens
            
            # 2. Response and reasoning length evolution
            response_lengths = []
            reasoning_lengths = []
            for iter in iterations:
                resp = iter.get('response', iter.get('full_response', ''))
                response_lengths.append(len(resp))
                reasoning = iter.get('reasoning', '')
                reasoning_lengths.append(len(reasoning))
            
            fig_lengths = go.Figure()
            fig_lengths.add_trace(go.Scatter(
                x=iteration_nums,
                y=response_lengths,
                mode='lines+markers',
                name='Response Length',
                line=dict(color='#27ae60', width=2)
            ))
            fig_lengths.add_trace(go.Scatter(
                x=iteration_nums,
                y=reasoning_lengths,
                mode='lines+markers',
                name='Reasoning Length',
                line=dict(color='#e67e22', width=2)
            ))
            fig_lengths.update_layout(
                title="Content Length Evolution",
                xaxis_title="Iteration",
                yaxis_title="Character Count",
                height=400,
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            charts['length_evolution'] = fig_lengths
            
            # 3. Similarity convergence chart
            similarities = []
            for i, iter_data in enumerate(iterations):
                if i == 0:
                    similarities.append(0.0)
                else:
                    sim = iter_data.get('similarity_to_previous', None)
                    if sim is not None:
                        similarities.append(sim)
            
            if len(similarities) == len(iterations):
                fig_similarity = go.Figure()
                fig_similarity.add_trace(go.Scatter(
                    x=iteration_nums,
                    y=similarities,
                    mode='lines+markers',
                    name='Similarity',
                    line=dict(color='#8e44ad', width=2),
                    marker=dict(size=8),
                    fill='tozeroy',
                    fillcolor='rgba(142, 68, 173, 0.1)'
                ))
                # Add convergence threshold line
                fig_similarity.add_hline(
                    y=0.99, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Convergence Threshold"
                )
                fig_similarity.update_layout(
                    title=f"Similarity Convergence ({similarity_mode.replace('_', ' ').title()})",
                    xaxis_title="Iteration",
                    yaxis_title="Similarity Score",
                    yaxis=dict(range=[0, 1.05]),
                    height=400,
                    showlegend=False,
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                charts['similarity_convergence'] = fig_similarity
            
            # 4. Similarity Heatmap (from visualizations module)
            try:
                fig_heatmap = create_similarity_heatmap(iterations)
                fig_heatmap.update_layout(
                    height=400,
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                charts['similarity_heatmap'] = fig_heatmap
            except Exception:
                pass
            
            # 5. Confidence Tracking
            try:
                fig_confidence = create_confidence_tracking(iterations)
                fig_confidence.update_layout(
                    height=400,
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                charts['confidence'] = fig_confidence
            except Exception:
                pass
            
            # 6. Complexity Metrics
            try:
                fig_complexity = create_complexity_metrics(iterations)
                fig_complexity.update_layout(
                    height=400,
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                charts['complexity'] = fig_complexity
            except Exception:
                pass
            
            # 7. Topic Flow Sankey
            try:
                if len(iterations) >= 2:
                    fig_topic = create_topic_flow_sankey(iterations)
                    fig_topic.update_layout(
                        height=400,
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )
                    charts['topic_flow'] = fig_topic
            except Exception:
                pass
            
            # 8. Divergence-Convergence Timeline
            try:
                fig_divergence = create_divergence_convergence_timeline(iterations)
                fig_divergence.update_layout(
                    height=400,
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                charts['divergence'] = fig_divergence
            except Exception:
                pass
                
    except Exception as e:
        print(f"Error creating charts: {e}")
    
    return charts


def create_pdf_report(thinking_loops: List[Dict], visualizations: Optional[Dict] = None, 
                     total_tokens: int = 0, total_time: float = 0, 
                     similarity_mode: str = "response_only") -> bytes:
    """
    Create a professional PDF report with main content and appendix
    
    Args:
        thinking_loops: List of thinking loop experiments
        visualizations: Dictionary containing visualization data
        total_tokens: Total tokens used across all experiments
        total_time: Total time elapsed
        similarity_mode: Mode used for similarity comparison
    
    Returns:
        PDF file as bytes
    """
    
    # Create a BytesIO buffer for the PDF
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
        title="LLM Reflection Lab Report",
        author="LLM Reflection Lab"
    )
    
    # Get and customize styles
    styles = getSampleStyleSheet()
    
    # Add custom styles (check if they already exist first)
    if 'CustomTitle' not in styles:
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
    
    if 'SectionTitle' not in styles:
        styles.add(ParagraphStyle(
            name='SectionTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=12,
            spaceBefore=20,
            borderWidth=1,
            borderColor=colors.HexColor('#3498db'),
            borderPadding=5
        ))
    
    if 'ExperimentTitle' not in styles:
        styles.add(ParagraphStyle(
            name='ExperimentTitle',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=10,
            spaceBefore=15,
            leftIndent=0,
            backColor=colors.HexColor('#ecf0f1')
        ))
    
    # Use BulletStyle to avoid conflict with existing Bullet style
    if 'BulletStyle' not in styles:
        styles.add(ParagraphStyle(
            name='BulletStyle',
            parent=styles['Normal'],
            leftIndent=20,
            bulletIndent=10
        ))
    
    if 'CodeStyle' not in styles:
        styles.add(ParagraphStyle(
            name='CodeStyle',
            parent=styles['Normal'],
            fontName='Courier',
            fontSize=9,
            leftIndent=10,
            rightIndent=10,
            spaceBefore=6,
            spaceAfter=6,
            backColor=colors.HexColor('#f5f5f5')
        ))
    
    if 'ReasoningStyle' not in styles:
        styles.add(ParagraphStyle(
            name='ReasoningStyle',
            parent=styles['Normal'],
            fontSize=10,
            leftIndent=10,
            rightIndent=10,
            spaceBefore=6,
            spaceAfter=6,
            backColor=colors.HexColor('#e8f6f3'),
            borderWidth=0.5,
            borderColor=colors.HexColor('#a6d5cf'),
            borderPadding=8
        ))
    
    if 'ResponseStyle' not in styles:
        styles.add(ParagraphStyle(
            name='ResponseStyle',
            parent=styles['Normal'],
            fontSize=10,
            leftIndent=10,
            rightIndent=10,
            spaceBefore=6,
            spaceAfter=6,
            backColor=colors.HexColor('#fef9e7'),
            borderWidth=0.5,
            borderColor=colors.HexColor('#f7dc6f'),
            borderPadding=8
        ))
    
    # Build the story (content)
    story = []
    
    # Title Page with enhanced colors
    story.append(Spacer(1, 0.5*inch))
    
    # Main title with gradient effect (simulated with colored background)
    title_style = ParagraphStyle(
        'TitlePage',
        parent=styles['Title'],
        fontSize=32,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=20,
        alignment=TA_CENTER,
        leading=40
    )
    story.append(Paragraph("🧠 LLM Reflection Lab", title_style))
    
    # Subtitle
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=colors.HexColor('#3498db'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    story.append(Paragraph("Iterative Thinking Analysis Report", subtitle_style))
    
    # Add decorative line
    line_data = [['']]
    line_table = Table(line_data, colWidths=[6*inch])
    line_table.setStyle(TableStyle([
        ('LINEBELOW', (0, 0), (-1, -1), 2, colors.HexColor('#3498db')),
    ]))
    story.append(line_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Add the main question if available
    if thinking_loops and len(thinking_loops) > 0 and thinking_loops[0].get('question'):
        question_style = ParagraphStyle(
            'QuestionStyle',
            parent=styles['Normal'],
            fontSize=14,
            textColor=colors.HexColor('#2c3e50'),
            alignment=TA_CENTER,
            leftIndent=30,
            rightIndent=30,
            spaceAfter=20,
            leading=18,
            backColor=colors.HexColor('#ecf6fd'),
            borderColor=colors.HexColor('#3498db'),
            borderWidth=1,
            borderPadding=15
        )
        
        # Get the first/main question
        main_question = thinking_loops[0].get('question', '')
        if 'Heading3' in styles:
            story.append(Paragraph(f"<b>Research Question:</b>", styles['Heading3']))
        else:
            story.append(Paragraph(f"<b>Research Question:</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(f"<i>{clean_text_for_pdf(main_question)}</i>", question_style))
        story.append(Spacer(1, 0.3*inch))
    
    # Report metadata with colorful table
    metadata_data = [
        ['📅 Report Generated:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ['🔬 Total Experiments:', str(len(thinking_loops))],
        ['🪙 Total Tokens Used:', f"{total_tokens:,}"],
        ['⏱️ Total Processing Time:', f"{total_time:.2f} seconds"],
        ['🎯 Similarity Mode:', similarity_mode.replace('_', ' ').title()]
    ]
    
    metadata_table = Table(metadata_data, colWidths=[2.5*inch, 3.5*inch])
    metadata_table.setStyle(TableStyle([
        # Header column styling
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
        # Value column styling
        ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#2c3e50')),
        # General styling
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.white),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        # Rounded corners effect with thicker border
        ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#3498db')),
    ]))
    story.append(metadata_table)
    
    story.append(PageBreak())
    
    # Executive Summary with colored header
    story.append(Paragraph("📊 Executive Summary", styles['SectionTitle']))
    story.append(Spacer(1, 0.2*inch))
    
    if thinking_loops:
        # Calculate summary statistics
        total_iterations = sum(len(loop.get('iterations', [])) for loop in thinking_loops)
        avg_iterations = total_iterations / len(thinking_loops) if thinking_loops else 0
        
        # Find convergence statistics if YOLO mode was used
        converged_experiments = [loop for loop in thinking_loops if loop.get('converged_early', False)]
        
        summary_text = f"""
        This report analyzes {len(thinking_loops)} experiment{'s' if len(thinking_loops) > 1 else ''} 
        conducted using the LLM Reflection Lab, with a total of {total_iterations} iterations performed.
        The experiments utilized {total_tokens:,} tokens over {total_time:.2f} seconds of processing time.
        """
        
        if converged_experiments:
            avg_convergence = sum(loop.get('convergence_iteration', 0) for loop in converged_experiments) / len(converged_experiments)
            summary_text += f"""
            
            YOLO Mode was used in {len(converged_experiments)} experiment{'s' if len(converged_experiments) > 1 else ''}, 
            with an average convergence at iteration {avg_convergence:.1f}.
            """
        
        for para in format_markdown_for_pdf(summary_text, styles):
            story.append(para)
    
    story.append(Spacer(1, 0.3*inch))
    
    # Key Findings Section
    story.append(Paragraph("Key Findings", styles['SectionTitle']))
    story.append(Spacer(1, 0.2*inch))
    
    # Limit to first 3 experiments or available experiments
    max_experiments = min(3, len(thinking_loops))
    for exp_num in range(max_experiments):
        loop = thinking_loops[exp_num]
        if loop.get('iterations') and len(loop['iterations']) > 0:
            final_iteration = loop['iterations'][-1]
            question_preview = loop.get('question', 'N/A')[:100]
            if len(loop.get('question', '')) > 100:
                question_preview += "..."
            
            story.append(Paragraph(f"<b>Experiment {exp_num + 1}:</b> {clean_text_for_pdf(question_preview)}", styles['Normal']))
            
            # Get final response
            final_response = final_iteration.get('response', final_iteration.get('full_response', 'No response'))
            response_preview = clean_text_for_pdf(final_response[:300])
            if len(final_response) > 300:
                response_preview += "..."
            
            story.append(Paragraph(f"<i>Final Response:</i> {response_preview}", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
    
    # Add visualizations section
    story.append(PageBreak())
    story.append(Paragraph("📊 Visualizations & Analytics", styles['SectionTitle']))
    story.append(Spacer(1, 0.2*inch))
    
    # Generate charts for all experiments combined
    all_iterations = []
    for loop in thinking_loops:
        if loop and loop.get('iterations'):
            all_iterations.extend(loop.get('iterations', []))
    
    if all_iterations and len(all_iterations) > 0:
        charts = create_visualization_charts(all_iterations, similarity_mode)
        
        # Page 1: Basic Metrics
        story.append(Paragraph("<b>Performance Metrics</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        # Add token usage and length evolution side by side
        chart_table_data = []
        row_charts = []
        
        if 'token_usage' in charts:
            row_charts.append(ChartFlowable(charts['token_usage'], width=3.5*inch, height=2.5*inch))
        
        if 'length_evolution' in charts:
            row_charts.append(ChartFlowable(charts['length_evolution'], width=3.5*inch, height=2.5*inch))
        
        if row_charts:
            if len(row_charts) == 1:
                story.append(row_charts[0])
            else:
                chart_table_data = [row_charts]
                chart_table = Table(chart_table_data)
                story.append(chart_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Page 2: Convergence Analysis
        story.append(Paragraph("<b>Convergence Analysis</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        row_charts = []
        if 'similarity_convergence' in charts:
            row_charts.append(ChartFlowable(charts['similarity_convergence'], width=3.5*inch, height=2.5*inch))
        
        if 'divergence' in charts:
            row_charts.append(ChartFlowable(charts['divergence'], width=3.5*inch, height=2.5*inch))
        
        if row_charts:
            if len(row_charts) == 1:
                story.append(row_charts[0])
            else:
                chart_table_data = [row_charts]
                chart_table = Table(chart_table_data)
                story.append(chart_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Similarity Heatmap (full width)
        if 'similarity_heatmap' in charts:
            story.append(Paragraph("<b>Iteration Similarity Matrix</b>", styles['Heading3']))
            story.append(ChartFlowable(charts['similarity_heatmap'], width=6*inch, height=4*inch))
            story.append(Spacer(1, 0.2*inch))
        
        # Page 3: Content Analysis
        story.append(PageBreak())
        story.append(Paragraph("<b>Content Analysis</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        row_charts = []
        if 'confidence' in charts:
            row_charts.append(ChartFlowable(charts['confidence'], width=3.5*inch, height=2.5*inch))
        
        if 'complexity' in charts:
            row_charts.append(ChartFlowable(charts['complexity'], width=3.5*inch, height=2.5*inch))
        
        if row_charts:
            if len(row_charts) == 1:
                story.append(row_charts[0])
            else:
                chart_table_data = [row_charts]
                chart_table = Table(chart_table_data)
                story.append(chart_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Topic Flow (full width)
        if 'topic_flow' in charts:
            story.append(Paragraph("<b>Topic Flow Analysis</b>", styles['Heading3']))
            story.append(ChartFlowable(charts['topic_flow'], width=6*inch, height=4*inch))
            story.append(Spacer(1, 0.2*inch))
        
        # Add per-experiment analysis if multiple experiments
        if len(thinking_loops) > 1:
            story.append(PageBreak())
            story.append(Paragraph("Per-Experiment Analysis", styles['Heading2']))
            
            for exp_num, loop in enumerate(thinking_loops, 1):
                if loop.get('iterations'):
                    story.append(Paragraph(f"<b>Experiment {exp_num}:</b> {clean_text_for_pdf(loop.get('question', 'N/A')[:50])}", styles['Heading3']))
                    exp_charts = create_visualization_charts(loop['iterations'], similarity_mode)
                    
                    # Create a compact view with 2x2 grid
                    chart_grid = []
                    row = []
                    
                    for chart_key in ['token_usage', 'similarity_convergence', 'confidence', 'complexity']:
                        if chart_key in exp_charts:
                            row.append(ChartFlowable(exp_charts[chart_key], width=3*inch, height=2*inch))
                            if len(row) == 2:
                                chart_grid.append(row)
                                row = []
                    
                    # Add remaining charts
                    if row:
                        while len(row) < 2:
                            row.append(Spacer(3*inch, 2*inch))
                        chart_grid.append(row)
                    
                    if chart_grid:
                        chart_table = Table(chart_grid)
                        story.append(chart_table)
                        story.append(Spacer(1, 0.2*inch))
    
    story.append(PageBreak())
    
    # Main Content - Detailed Experiments
    story.append(Paragraph("Detailed Experiment Results", styles['SectionTitle']))
    story.append(Spacer(1, 0.2*inch))
    
    for exp_num, loop in enumerate(thinking_loops, 1):
        # Experiment header
        story.append(Paragraph(f"Experiment {exp_num}", styles['ExperimentTitle']))
        
        # Experiment metadata with enhanced colors
        exp_data = [
            ['🔍 Question:', clean_text_for_pdf(loop.get('question', 'N/A'))],
            ['📅 Timestamp:', loop.get('timestamp', 'N/A')[:19]],
            ['🔄 Iterations:', str(len(loop.get('iterations', [])))],
            ['🪙 Tokens Used:', f"{loop.get('tokens_used', 0):,}"],
        ]
        
        if loop.get('converged_early'):
            exp_data.append(['✅ Convergence:', f"Iteration {loop.get('convergence_iteration', 'N/A')}"])
            exp_data.append(['📊 Final Similarity:', f"{loop.get('final_similarity', 0):.3f}"])
        
        exp_table = Table(exp_data, colWidths=[1.5*inch, 4.5*inch])
        
        # Build styles dynamically based on number of rows
        table_styles = [
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            # Font styling
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.white),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#3498db')),
        ]
        
        # Add alternating row colors for existing rows only
        for i in range(1, len(exp_data)):
            if i < 4:
                # Regular rows with alternating colors
                color = colors.HexColor('#e8f4f8') if i % 2 == 1 else colors.HexColor('#f0f8fc')
            else:
                # Convergence rows with green tint
                color = colors.HexColor('#d4f1d4') if i % 2 == 0 else colors.HexColor('#e8f8e8')
            table_styles.append(('BACKGROUND', (0, i), (-1, i), color))
        
        exp_table.setStyle(TableStyle(table_styles))
        story.append(exp_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Show final iteration details
        if loop.get('iterations'):
            final_iteration = loop['iterations'][-1]
            
            story.append(Paragraph("<b>Final Reasoning:</b>", styles['Normal']))
            reasoning_text = final_iteration.get('reasoning', 'No reasoning captured')
            for para in format_markdown_for_pdf(reasoning_text[:500], styles):  # Limit for main section
                story.append(para)
            if len(reasoning_text) > 500:
                story.append(Paragraph("<i>... (see appendix for complete text)</i>", styles['Normal']))
            
            story.append(Spacer(1, 0.1*inch))
            
            story.append(Paragraph("<b>Final Response:</b>", styles['Normal']))
            response_text = final_iteration.get('response', final_iteration.get('full_response', 'No response'))
            for para in format_markdown_for_pdf(response_text[:500], styles):  # Limit for main section
                story.append(para)
            if len(response_text) > 500:
                story.append(Paragraph("<i>... (see appendix for complete text)</i>", styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Add page break between experiments
        if exp_num < len(thinking_loops):
            story.append(PageBreak())
    
    # Appendix - All Iterations
    story.append(PageBreak())
    story.append(Paragraph("Appendix: Complete Iteration Details", styles['SectionTitle']))
    story.append(Spacer(1, 0.2*inch))
    
    for exp_num, loop in enumerate(thinking_loops, 1):
        story.append(Paragraph(f"Experiment {exp_num} - All Iterations", styles['ExperimentTitle']))
        story.append(Paragraph(f"<i>Question: {clean_text_for_pdf(loop.get('question', 'N/A'))}</i>", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        
        for iteration in loop.get('iterations', []):
            # Create a keep-together group for each iteration
            iteration_content = []
            
            iteration_content.append(Paragraph(
                f"<b>Iteration {iteration.get('iteration_number', 'N/A')}</b> - "
                f"{iteration.get('timestamp', 'N/A')[:19]} - "
                f"{iteration.get('usage', {}).get('total_tokens', 0):,} tokens",
                styles['Normal']
            ))
            
            if iteration.get('similarity_to_previous') is not None:
                iteration_content.append(Paragraph(
                    f"<i>Similarity to previous: {iteration['similarity_to_previous']:.3f}</i>",
                    styles['Normal']
                ))
            
            iteration_content.append(Spacer(1, 0.05*inch))
            
            # Reasoning
            iteration_content.append(Paragraph("<b>Reasoning:</b>", styles['Normal']))
            reasoning_text = iteration.get('reasoning', 'No reasoning captured')
            for para in format_markdown_for_pdf(reasoning_text, styles):
                iteration_content.append(para)
            
            iteration_content.append(Spacer(1, 0.05*inch))
            
            # Response
            iteration_content.append(Paragraph("<b>Response:</b>", styles['Normal']))
            response_text = iteration.get('response', iteration.get('full_response', 'No response'))
            for para in format_markdown_for_pdf(response_text, styles):
                iteration_content.append(para)
            
            iteration_content.append(Spacer(1, 0.2*inch))
            
            # Try to keep iteration together on same page
            story.append(KeepTogether(iteration_content))
        
        # Page break between experiments in appendix
        if exp_num < len(thinking_loops):
            story.append(PageBreak())
    
    # Build the PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    
    # Get the PDF bytes
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes


def export_to_pdf(thinking_loops: List[Dict], visualizations: Optional[Dict] = None,
                 total_tokens: int = 0, total_time: float = 0,
                 similarity_mode: str = "response_only", llm_client=None) -> tuple[bytes, str]:
    """
    Main export function to create PDF report
    
    Args:
        thinking_loops: List of thinking loop experiments
        visualizations: Optional visualization data
        total_tokens: Total tokens used
        total_time: Total processing time
        similarity_mode: Mode used for similarity comparison
        llm_client: Optional LLM client for generating smart filename
    
    Returns:
        Tuple of (PDF file as bytes, suggested filename)
    """
    try:
        pdf_bytes = create_pdf_report(
            thinking_loops=thinking_loops,
            visualizations=visualizations,
            total_tokens=total_tokens,
            total_time=total_time,
            similarity_mode=similarity_mode
        )
    except Exception as e:
        # If PDF generation fails, raise with more context
        raise Exception(f"PDF generation failed: {str(e)}")
    
    # Generate smart filename based on question using the same LLM
    filename = 'llm-reflection-report'
    try:
        if thinking_loops and len(thinking_loops) > 0 and thinking_loops[0].get('question'):
            filename = generate_filename_from_question(thinking_loops[0]['question'], llm_client)
    except Exception:
        # If filename generation fails, use default
        pass
    
    return pdf_bytes, filename