"""Code generation utilities — extract code and apply diffs from LLM responses.

Vault: Implementation Details — Diff-Based Generation.md

Two modes:
- Full: LLM returns complete code in a fenced block
- Diff: LLM returns SEARCH/REPLACE blocks applied to existing code
"""

import re
from typing import List, Tuple


def extract_code(text: str) -> str:
    """Extract code from an LLM response.

    Looks for fenced code blocks (```python or ```). Returns the last block
    found (LLMs often refine across multiple blocks). If no blocks, returns
    the full text stripped.
    """
    blocks = re.findall(r'```python\s*\n(.*?)```', text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()

    blocks = re.findall(r'```\s*\n(.*?)```', text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()

    return text.strip()


def parse_diff(text: str) -> List[Tuple[str, str]]:
    """Parse SEARCH/REPLACE blocks from text.

    Format:
        <<<<<<< SEARCH
        old code
        =======
        new code
        >>>>>>> REPLACE

    Returns list of (search, replace) tuples.
    """
    pattern = r'<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches

    pattern = r'<{3,}\s*SEARCH\s*\n(.*?)\n={3,}\s*\n(.*?)\n>{3,}\s*REPLACE'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def apply_diff(code: str, diffs: List[Tuple[str, str]]) -> str:
    """Apply SEARCH/REPLACE blocks to code.

    Each (search, replace) is applied in order. Raises ValueError if a
    search string is not found in the code.
    """
    for search, replace in diffs:
        if search not in code:
            search_stripped = "\n".join(line.rstrip() for line in search.split("\n"))
            code_stripped = "\n".join(line.rstrip() for line in code.split("\n"))
            if search_stripped in code_stripped:
                code = code.replace(search_stripped, replace)
                continue
            raise ValueError(f"Search block not found in code:\n{search[:100]}...")
        code = code.replace(search, replace, 1)
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
