"""PDF Export Module for LLM Reflection Lab"""

import io
import base64
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.io as pio
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
    """Convert markdown-like text to ReportLab Paragraph objects"""
    if not text:
        return [Paragraph("No content", styles['Normal'])]
    
    # Clean the text first
    text = clean_text_for_pdf(text)
    
    paragraphs = []
    lines = text.split('\n')
    current_para = []
    in_code_block = False
    
    for line in lines:
        # Handle code blocks
        if line.strip().startswith('```'):
            if current_para:
                para_text = ' '.join(current_para)
                if in_code_block:
                    paragraphs.append(Paragraph(f'<pre>{para_text}</pre>', styles.get('CodeStyle', styles['Normal'])))
                else:
                    paragraphs.append(Paragraph(para_text, styles['Normal']))
                current_para = []
            in_code_block = not in_code_block
            continue
        
        # Handle headers
        if line.startswith('### '):
            if current_para:
                paragraphs.append(Paragraph(' '.join(current_para), styles['Normal']))
                current_para = []
            if 'Heading3' in styles:
                paragraphs.append(Paragraph(line[4:], styles['Heading3']))
            else:
                paragraphs.append(Paragraph(line[4:], styles['Heading2']))
            continue
        elif line.startswith('## '):
            if current_para:
                paragraphs.append(Paragraph(' '.join(current_para), styles['Normal']))
                current_para = []
            paragraphs.append(Paragraph(line[3:], styles['Heading2']))
            continue
        elif line.startswith('# '):
            if current_para:
                paragraphs.append(Paragraph(' '.join(current_para), styles['Normal']))
                current_para = []
            paragraphs.append(Paragraph(line[2:], styles['Heading1']))
            continue
        
        # Handle lists
        if line.strip().startswith('- ') or line.strip().startswith('* '):
            if current_para:
                paragraphs.append(Paragraph(' '.join(current_para), styles['Normal']))
                current_para = []
            list_text = line.strip()[2:] if len(line.strip()) > 2 else ''
            paragraphs.append(Paragraph(f"• {list_text}", styles.get('BulletStyle', styles['Normal'])))
            continue
        
        # Handle numbered lists
        import re
        if re.match(r'^\d+\.\s', line.strip()):
            if current_para:
                paragraphs.append(Paragraph(' '.join(current_para), styles['Normal']))
                current_para = []
            paragraphs.append(Paragraph(line.strip(), styles.get('BulletStyle', styles['Normal'])))
            continue
        
        # Regular text
        if line.strip():
            # Apply inline formatting safely
            try:
                # Count occurrences to ensure proper pairing
                bold_count = line.count('**')
                if bold_count >= 2:
                    line = line.replace('**', '<b>', 1).replace('**', '</b>', 1)
                
                italic_count = line.count('*')
                if italic_count >= 2:
                    line = line.replace('*', '<i>', 1).replace('*', '</i>', 1)
                
                code_count = line.count('`')
                if code_count >= 2:
                    line = line.replace('`', '<font face="Courier">', 1).replace('`', '</font>', 1)
            except:
                pass  # If formatting fails, use plain text
            
            current_para.append(line)
        elif current_para:
            # Empty line indicates paragraph break
            para_text = ' '.join(current_para)
            if in_code_block:
                paragraphs.append(Paragraph(f'<pre>{para_text}</pre>', styles.get('CodeStyle', styles['Normal'])))
            else:
                paragraphs.append(Paragraph(para_text, styles['Normal']))
            current_para = []
    
    # Don't forget the last paragraph
    if current_para:
        para_text = ' '.join(current_para)
        if in_code_block:
            paragraphs.append(Paragraph(f'<pre>{para_text}</pre>', styles.get('CodeStyle', styles['Normal'])))
        else:
            paragraphs.append(Paragraph(para_text, styles['Normal']))
    
    return paragraphs if paragraphs else [Paragraph("No content", styles['Normal'])]


def create_visualization_charts(iterations: List[Dict], similarity_mode: str = "response_only") -> Dict:
    """Create visualization charts for the PDF report"""
    charts = {}
    
    try:
        # Token usage over time chart
        if iterations:
            token_counts = [iter.get('usage', {}).get('total_tokens', 0) for iter in iterations]
            iteration_nums = list(range(1, len(iterations) + 1))
            
            fig_tokens = go.Figure()
            fig_tokens.add_trace(go.Scatter(
                x=iteration_nums,
                y=token_counts,
                mode='lines+markers',
                name='Tokens Used',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ))
            fig_tokens.update_layout(
                title="Token Usage Per Iteration",
                xaxis_title="Iteration",
                yaxis_title="Tokens",
                height=400,
                showlegend=False
            )
            charts['token_usage'] = fig_tokens
            
            # Response length evolution
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
                line=dict(color='green', width=2)
            ))
            fig_lengths.add_trace(go.Scatter(
                x=iteration_nums,
                y=reasoning_lengths,
                mode='lines+markers',
                name='Reasoning Length',
                line=dict(color='orange', width=2)
            ))
            fig_lengths.update_layout(
                title="Content Length Evolution",
                xaxis_title="Iteration",
                yaxis_title="Character Count",
                height=400,
                showlegend=True
            )
            charts['length_evolution'] = fig_lengths
            
            # Similarity convergence chart (if similarity data exists)
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
                    line=dict(color='purple', width=2),
                    marker=dict(size=8)
                ))
                fig_similarity.update_layout(
                    title=f"Similarity Convergence ({similarity_mode.replace('_', ' ').title()})",
                    xaxis_title="Iteration",
                    yaxis_title="Similarity Score",
                    yaxis=dict(range=[0, 1.05]),
                    height=400,
                    showlegend=False
                )
                charts['similarity'] = fig_similarity
                
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
    story.append(Paragraph("Visualizations & Analytics", styles['SectionTitle']))
    story.append(Spacer(1, 0.2*inch))
    
    # Generate charts for all experiments combined
    all_iterations = []
    for loop in thinking_loops:
        if loop and loop.get('iterations'):
            all_iterations.extend(loop.get('iterations', []))
    
    if all_iterations and len(all_iterations) > 0:
        charts = create_visualization_charts(all_iterations, similarity_mode)
        
        # Add each chart to the story
        if 'token_usage' in charts:
            story.append(Paragraph("Token Usage Analysis", styles['Heading3']))
            story.append(ChartFlowable(charts['token_usage'], width=6*inch, height=3*inch))
            story.append(Spacer(1, 0.2*inch))
        
        if 'length_evolution' in charts:
            story.append(Paragraph("Content Length Evolution", styles['Heading3']))
            story.append(ChartFlowable(charts['length_evolution'], width=6*inch, height=3*inch))
            story.append(Spacer(1, 0.2*inch))
        
        if 'similarity' in charts:
            story.append(Paragraph("Convergence Analysis", styles['Heading3']))
            story.append(ChartFlowable(charts['similarity'], width=6*inch, height=3*inch))
            story.append(Spacer(1, 0.2*inch))
        
        # Add individual experiment charts if multiple experiments
        if len(thinking_loops) > 1:
            story.append(PageBreak())
            story.append(Paragraph("Per-Experiment Analysis", styles['Heading2']))
            
            for exp_num, loop in enumerate(thinking_loops, 1):
                if loop.get('iterations'):
                    story.append(Paragraph(f"Experiment {exp_num}", styles['Heading3']))
                    exp_charts = create_visualization_charts(loop['iterations'], similarity_mode)
                    
                    if 'token_usage' in exp_charts:
                        story.append(ChartFlowable(exp_charts['token_usage'], width=5*inch, height=2.5*inch))
                        story.append(Spacer(1, 0.1*inch))
    
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
        exp_table.setStyle(TableStyle([
            # Alternating row colors
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#e8f4f8')),
            ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#f0f8fc')),
            ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#e8f4f8')),
            # Convergence rows if present
            ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#d4f1d4')),
            ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor('#e8f8e8')),
            # Font styling
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.white),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#3498db')),
        ]))
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