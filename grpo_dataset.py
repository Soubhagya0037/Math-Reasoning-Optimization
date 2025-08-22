import re
import datasets
from datasets import load_dataset, Dataset

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_the_answer_is(text: str) -> str | None:
    """
    Extracts the final numeric answer from text after 'The answer is:'
    Returns None if not found.
    """
    match = re.search(r"The answer is:\s*([-+]?(?:[\d.a-zA-Z\\{}\[\]()\/^+\-* ]+))$", text)
    if match:
        return match.group(1)
    return None

# uncomment middle messages for 1-shot prompting
def get_questions(split = "train") -> Dataset:
    data = datasets.load_dataset("json", data_files="data/metamath_10k.jsonl", split="train")
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_the_answer_is(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_questions()
dataset.to_json("try_grpo_dataset.json")