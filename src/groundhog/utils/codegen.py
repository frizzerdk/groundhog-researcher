"""Code generation utilities — extract code and apply diffs from LLM responses.

Data flow:
    response = backend.generate(prompt, system_prompt)   # raw LLM text
    new_code, diff = extract_code(response.text, prior)  # valid Python or ""

extract_code tries in order:
    1. SEARCH/REPLACE diffs applied to prior_code
    2. Fenced ```python block
    3. Fenced ``` block (no language tag)
    4. Raw text with fence cleanup
Each candidate validated with compile(). Returns ("", Diff("none")) if nothing valid.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Diff:
    """Metadata about how code was extracted from an LLM response."""
    method: str = "none"    # "search_replace", "fenced", "raw", "none"
    blocks: int = 0         # number of diff blocks applied


def extract_code(text: str, prior_code: str = None) -> Tuple[str, Diff]:
    """Extract valid Python code from an LLM response.

    Args:
        text: raw LLM response text
        prior_code: existing code to apply diffs against (None or "" for fresh)

    Returns:
        (code, diff) — code is valid Python or "", diff has extraction metadata
    """
    prior = prior_code or ""

    # 1. SEARCH/REPLACE diffs (only if we have prior code to apply them to)
    if prior:
        diffs = _parse_diff(text)
        if diffs:
            try:
                result = _apply_diff(prior, diffs)
                compile(result, "<string>", "exec")
                return result, Diff("search_replace", blocks=len(diffs))
            except (ValueError, SyntaxError):
                pass  # fall through to other methods

    # 2. Fenced ```python block
    blocks = re.findall(r'```python\s*\n(.*?)```', text, re.DOTALL)
    if blocks:
        candidate = blocks[-1].strip()
        try:
            compile(candidate, "<string>", "exec")
            return candidate, Diff("fenced")
        except SyntaxError:
            pass

    # 3. Fenced ``` block (no language tag)
    blocks = re.findall(r'```\s*\n(.*?)```', text, re.DOTALL)
    if blocks:
        candidate = blocks[-1].strip()
        try:
            compile(candidate, "<string>", "exec")
            return candidate, Diff("fenced")
        except SyntaxError:
            pass

    # 4. Raw text — strip stray fence markers, try compile
    cleaned = re.sub(r'^```\w*\s*\n?', '', text)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned)
    cleaned = cleaned.strip()
    if cleaned:
        try:
            compile(cleaned, "<string>", "exec")
            return cleaned, Diff("raw")
        except SyntaxError:
            pass

    # 5. Nothing valid
    return "", Diff("none")


# --- Internal helpers (used by extract_code) ---

def _parse_diff(text: str) -> List[Tuple[str, str]]:
    """Parse SEARCH/REPLACE blocks from text."""
    pattern = r'<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches
    # Relaxed pattern (variable marker lengths)
    pattern = r'<{3,}\s*SEARCH\s*\n(.*?)\n={3,}\s*\n(.*?)\n>{3,}\s*REPLACE'
    return re.findall(pattern, text, re.DOTALL)


def _apply_diff(code: str, diffs: List[Tuple[str, str]]) -> str:
    """Apply SEARCH/REPLACE blocks to code. Raises ValueError if search not found."""
    for search, replace in diffs:
        if search in code:
            code = code.replace(search, replace, 1)
            continue
        # Try with trailing whitespace stripped
        search_stripped = "\n".join(line.rstrip() for line in search.split("\n"))
        code_stripped = "\n".join(line.rstrip() for line in code.split("\n"))
        if search_stripped in code_stripped:
            code = code.replace(search_stripped, replace)
            continue
        raise ValueError(f"Search block not found in code:\n{search[:100]}...")
    return code


def build_prompt(context: str, prior_code: str = None, learnings: str = None,
                 mode: str = "full") -> str:
    """Assemble a prompt for code generation.

    mode="full": ask LLM to write complete code
    mode="diff": ask LLM to write SEARCH/REPLACE blocks improving prior code
    """
    parts = []

    if mode == "diff":
        parts.append("Improve the following code using SEARCH/REPLACE blocks.")
        parts.append("Format each change as:")
        parts.append("<<<<<<< SEARCH\nold code\n=======\nnew code\n>>>>>>> REPLACE")
    else:
        parts.append("Write complete, runnable code in a ```python block.")

    parts.append(f"\n## Task\n{context}")

    if learnings:
        parts.append(f"\n## Learnings\n{learnings}")

    if prior_code:
        label = "Current code to improve" if mode == "diff" else "Reference (previous best)"
        parts.append(f"\n## {label}\n```python\n{prior_code}\n```")

    return "\n\n".join(parts)
