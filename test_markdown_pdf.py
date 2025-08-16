#!/usr/bin/env python3
"""Test script to verify markdown rendering in PDF"""

from src.pdf_export import export_to_pdf
import tempfile
import os

# Create test data with various markdown elements
test_iterations = [
    {
        "iteration": 1,
        "reasoning": "This is a test with **bold text**, *italic text*, and `inline code`.",
        "response": """
# Header 1
## Header 2
### Header 3

This paragraph has **bold**, *italic*, ***bold italic***, and `code` formatting.

- List item with **bold**
- List item with *italic*
- List item with `code`

1. Numbered list
2. With multiple items
3. And **formatting**

> This is a blockquote with *italic* text

```python
def test_function():
    return "Hello World"
```

Regular paragraph with [link text](http://example.com) that should display.
""",
        "confidence": 0.85,
        "token_usage": {"reasoning_tokens": 100, "response_tokens": 150, "total_tokens": 250}
    },
    {
        "iteration": 2,
        "reasoning": "Second iteration with more **markdown** elements.",
        "response": "Simple response with **bold** and *italic* and `code`.",
        "confidence": 0.90,
        "token_usage": {"reasoning_tokens": 50, "response_tokens": 75, "total_tokens": 125}
    }
]

# Generate PDF
print("Generating test PDF with markdown...")
try:
    pdf_data, filename = export_to_pdf(
        thinking_loops=test_iterations,
        visualizations=None,
        total_tokens=375,
        total_time=5.2,
        similarity_mode="response_only",
        llm_client=None
    )
    
    # Save to file
    import os
    output_path = os.path.join("saves", filename)
    os.makedirs("saves", exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(pdf_data)
    
    print(f"✅ PDF generated successfully: {filename}")
    print(f"   Path: {output_path}")
    print("\nPlease open the PDF to verify markdown rendering:")
    print(f"open '{output_path}'")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()