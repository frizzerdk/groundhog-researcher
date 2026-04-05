"""Default agent tools built from toolkit capabilities.

Uses agent_tool() factory to wrap toolkit methods as agent-callable tools.
The optimizer calls build_default_agent_tools() and puts the result on
toolkit.agent_tools. Users can extend the list with custom tools.

All tools are built at optimizer init time — no workspace binding needed.
The strategy controls which tools are available per phase via filtering.
"""

from groundhog.base.agent import agent_tool


def build_default_agent_tools(toolkit) -> list:
    """Build agent tools from toolkit capabilities.

    Inspects what's on the toolkit and wraps available methods.
    Returns a plain list — caller puts it on toolkit.agent_tools.
    """
    tools = []

    if hasattr(toolkit, 'learnings'):
        tools.append(agent_tool(
            name="get-learnings",
            description="Read accumulated learnings about what works and what doesn't for this task",
            func=toolkit.learnings.get,
            params={
                "last": {"type": "int", "default": 20, "description": "Number of recent entries"},
                "random": {"type": "int", "default": 10, "description": "Number of random older entries"},
            },
        ))

    if hasattr(toolkit, 'task'):
        through = getattr(toolkit, 'through', None)
        for stage in toolkit.task.evaluator.eval_stages(toolkit.task.data, through=through):
            tools.append(agent_tool(
                name=stage.name,
                description=f"{stage.description}. Pass the path to a .py file to evaluate.",
                func=lambda path, s=stage: s.call(open(path).read()),
                params={
                    "path": {"type": "str", "description": "Path to .py file to evaluate"},
                },
            ))

    return tools
