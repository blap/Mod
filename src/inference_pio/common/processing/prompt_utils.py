"""
Prompt Utilities for Qwen3 and Standardized Benchmarking.
Enforces prompt formatting rules for Math, MCQ, and Chat History.
"""

import json
from typing import List, Dict, Optional, Union

def format_math_prompt(query: str) -> str:
    """
    Formats a math problem prompt with the required reasoning instruction.
    Rule: Include "Please reason step by step, and put your final answer within \\boxed{}."
    """
    suffix = "Please reason step by step, and put your final answer within \\boxed{}."
    if suffix not in query:
        return f"{query}\n\n{suffix}"
    return query

def format_mcq_prompt(query: str) -> str:
    """
    Formats a multiple-choice question prompt with the required JSON output instruction.
    Rule: Add JSON structure instruction for standardized responses.
    """
    instruction = 'Please show your choice in the answer field with only the choice letter, e.g., "answer": "C".'
    if instruction not in query:
        return f"{query}\n\n{instruction}"
    return query

def format_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Processes chat history to strip 'thinking content' from model outputs.
    Rule: Historical model output should only include the final output part.

    Assumes thinking content is wrapped in <think>...</think> tags.
    """
    cleaned_history = []
    for turn in history:
        role = turn.get("role")
        content = turn.get("content", "")

        if role == "assistant" or role == "model":
            # Remove <think> content
            start_tag = "<think>"
            end_tag = "</think>"

            while start_tag in content and end_tag in content:
                start_idx = content.find(start_tag)
                end_idx = content.find(end_tag) + len(end_tag)
                if start_idx < end_idx:
                    content = content[:start_idx] + content[end_idx:]
                else:
                    break

            cleaned_history.append({"role": role, "content": content.strip()})
        else:
            cleaned_history.append(turn)

    return cleaned_history

def apply_chat_template(
    tokenizer,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
    enable_thinking: bool = True,
    tokenize: bool = False
) -> Union[str, List[int]]:
    """
    Simplified chat template application matching Hugging Face signature.
    Injects <think> tags if enabled for the current generation.
    """
    # 1. Format the text
    formatted_text = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        formatted_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    if add_generation_prompt:
        formatted_text += "<|im_start|>assistant\n"
        if enable_thinking:
            formatted_text += "<think>\n"

    # 2. Tokenize if requested
    if tokenize:
        if hasattr(tokenizer, 'encode'):
            return tokenizer.encode(formatted_text)
        else:
            # Fallback
            return [0] * 10

    return formatted_text
