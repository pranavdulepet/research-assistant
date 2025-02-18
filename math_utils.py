import re
from typing import List, Dict
import logging

class MathExpression:
    def __init__(self, expression: str, context: str, start_pos: int, end_pos: int):
        self.expression = expression
        self.context = context
        self.start_pos = start_pos
        self.end_pos = end_pos

def extract_math_expressions(text: str) -> List[MathExpression]:
    """Extract mathematical expressions from text using regex patterns"""
    math_patterns = [
        (r'\$.*?\$', 'inline_latex'),  # LaTeX inline math
        (r'\\\[.*?\\\]', 'display_latex'),  # LaTeX display math
        (r'\\begin\{equation\}.*?\\end\{equation\}', 'equation_latex'),  # LaTeX equation environment
        (r'(?<=\s)[a-zA-Z][=><+\-*/][0-9]+(?=\s)', 'simple_equation'),  # Simple equations
        (r'(?:\d+\s*[-+*/=]\s*)+\d+', 'arithmetic')  # Arithmetic expressions
    ]
    
    expressions = []
    for pattern, type_ in math_patterns:
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            context_start = max(0, match.start() - 200)
            context_end = min(len(text), match.end() + 200)
            
            expressions.append(MathExpression(
                expression=match.group(),
                context=text[context_start:context_end],
                start_pos=match.start(),
                end_pos=match.end()
            ))
    
    return expressions

def clean_math_expression(expr: str) -> str:
    """Clean and normalize mathematical expressions"""
    # Remove extra whitespace
    expr = ' '.join(expr.split())
    # Fix common OCR errors
    expr = expr.replace('ˆ', '^')
    expr = expr.replace('−', '-')
    return expr

def detect_math_expressions(text):
    """Detect mathematical expressions in text using various patterns."""
    patterns = [
        r'\$.*?\$',  # LaTeX inline math
        r'\\\[.*?\\\]',  # LaTeX display math
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',  # Variable assignments
        r'(?:\d+\s*[-+*/^]\s*)+\d+',  # Arithmetic expressions
        r'[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)',  # Function calls
        r'∑|∏|∫|∂|∇|∆|√',  # Mathematical symbols
        r'[a-zA-Z_][a-zA-Z0-9_]*(?:\s*[+\-*/^=<>≤≥]\s*[a-zA-Z0-9_]+)+',  # Complex expressions
    ]
    
    math_expressions = []
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            expression = match.group()
            context_start = max(0, match.start() - 100)
            context_end = min(len(text), match.end() + 100)
            
            math_expressions.append({
                'expression': expression,
                'context': text[context_start:context_end],
                'start': match.start(),
                'end': match.end()
            })
    
    return math_expressions

def format_math_expression(expression):
    """Format a mathematical expression for display."""
    # Convert basic operations to LaTeX
    expression = expression.replace('*', '\\times ')
    expression = expression.replace('/', '\\div ')
    expression = re.sub(r'\^(\d+)', r'^{\1}', expression)
    
    # Add LaTeX delimiters if not present
    if not expression.startswith('$'):
        expression = f'${expression}$'
    
    return expression
