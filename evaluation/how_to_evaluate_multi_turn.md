# Multi-Turn Evaluation with Pydantic AI

## Pattern

Chain `agent.run()` calls, passing `message_history` from each result into the next turn:

```python
from pydantic_ai import Agent

agent = Agent(model, instructions=SYSTEM_PROMPT)

# Predefined user messages for this evaluation scenario
user_messages = [
    "Here is the OEM loads file. Please process it.",
    "Yes, use envelope method for the limit loads.",
    "Looks good. Now generate the .inp files.",
]

# Turn 1
result = await agent.run(user_messages[0])

# Turn 2 — pass history from turn 1
result = await agent.run(
    user_messages[1],
    message_history=result.all_messages(),
)

# Turn 3 — pass accumulated history
result = await agent.run(
    user_messages[2],
    message_history=result.all_messages(),
)

# result.all_messages() now contains the full conversation
```

Each call to `result.all_messages()` returns the **entire** conversation so far (system prompt + all user/assistant turns), so you just keep threading it forward.

## Adapting the Evaluator

Replace the single `agent.run(prompt)` in `run_agent()` with a loop:

```python
async def run_agent(model, scenario) -> Solution:
    agent = Agent(model, output_type=str, instructions=SYSTEM_PROMPT)

    # Define the scripted user turns per scenario
    user_turns = scenario.user_messages  # list[str]

    history = None
    for msg in user_turns:
        result = await agent.run(msg, message_history=history or [])
        history = result.all_messages()

    return _parse_solution(result.output)
```

## Key Details

- `message_history=[]` (empty list) on the first turn generates a fresh system prompt
- `message_history=result.all_messages()` on subsequent turns reuses the existing system prompt (it won't generate a new one)
- The agent's tool calls and model responses from all turns are preserved in the history
- For evaluation, return the full `result` (not just `.output`) so evaluators can inspect `result.all_messages()` for the entire conversation

## With Pydantic Evals

Per the [GitHub discussion](https://github.com/pydantic/pydantic-ai/issues/3220), have your evaluated function return the full run result, then write custom evaluators that check `result.all_messages()`:

```python
from pydantic_evals import Case, Dataset

cases = [
    Case(
        name="scenario_v2_multiturn",
        inputs=["msg1", "msg2", "msg3"],  # predefined user messages
        expected_output=...,
        evaluators=[MethodologyEvaluator(), ArtifactEvaluator()],
    )
]
```

## References

- [Messages and chat history - Pydantic AI](https://ai.pydantic.dev/message-history/)
- [Multi-turn eval discussion (issue #3220)](https://github.com/pydantic/pydantic-ai/issues/3220)
- [Agent API docs](https://ai.pydantic.dev/agent/)
