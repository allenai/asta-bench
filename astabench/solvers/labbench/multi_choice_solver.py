from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ResponseSchema,
    get_model,
)
from inspect_ai.solver import Solver, solver, TaskState, Generate
from inspect_ai.util import json_schema
from pydantic import BaseModel

from astabench.evals.sqa.retry_utils import generate_with_retry


class MCQAnswer(BaseModel):
    answer: str


@solver
def multi_choice_solver(
        model: str | None = None, mcq_string: bool = True
) -> Solver:
    model = get_model(model)

    async def solve(state: TaskState, generate: Generate):
        m = state.messages[-1]
        if not isinstance(m, ChatMessageAssistant):
            raise ValueError("The last message must be from the assistant.")
        question = state.input
        response = state.messages[-1].text.replace(f"Question: {question}", "").split("References")[0].strip()
        choices = None
        if not mcq_string:
            mcq_prompt = """You are given a multiple choice question in <question></question> and a system response in <response></response> tags. 
            Along with the question and response, you are also given a list of possible answer choices in <choices></choices> tags. 
            Based on the question and given response, your task is to select the correct choice from the list of choices. 
            Output a json with the key "answer" and the value being only the single letter of the correct choice."""
            choices = state.choices
        else:
            mcq_prompt = """You are given a multiple choice question with a list of possible answer choices in <question></question> tags
            and a system response in <response></response> tags. 
            Based on the question and given response, your task is to select the correct choice from the list of choices."""

        mcq_prompt += """\nIf you cannot determine the correct choice from the
            question and answer, output the letter choice corresponding to 'Insufficient information'."""

        user_prompt = f"<question>{question}</question>\n\n<response>{response}</response>"
        if choices:
            choices_text = "\n".join(
                f"{chr(ord('A') + idx)}. {choice.value}"
                for idx, choice in enumerate(choices)
            )
            user_prompt += f"\n\n<choices>{choices_text}\n</choices>"

        mcq_messages = [
            ChatMessageSystem(content=mcq_prompt),
            ChatMessageUser(content=user_prompt),
        ]

        output_result, out_result_dict, _ = await generate_with_retry(
            model,
            mcq_messages,
            config=GenerateConfig(
                response_schema=ResponseSchema(
                    name="mcq_answer",
                    json_schema=json_schema(MCQAnswer),
                ),
                max_tokens=10,
            ),
        )
        ans_letter = out_result_dict.get("answer", "").strip().upper()
        ans_indx = ord(ans_letter) - ord("A")
        if not (0 <= ans_indx < len(state.choices)):
            raise ValueError(f"Invalid answer letter: {ans_letter}")
        # Mark the correct choice
        state.choices.mark_choice(ans_indx, True)
        return state

    return solve
