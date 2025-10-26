import hydra
from omegaconf import DictConfig

from loguru import logger

import json
from pathlib import Path
from dotenv import load_dotenv

from pydantic import BaseModel

from langchain_openai import ChatOpenAI

class Question(BaseModel):
    question: str
    answer: str
    category: str

class Questions(BaseModel):
    questions: list[Question]


def synthetic_data_generation(prompt, llm: ChatOpenAI):
    """
    Generate synthetic data using a language model with structured output.

    This function takes a prompt and uses a ChatOpenAI language model to generate
    structured synthetic data in the form of Questions objects.

    Args:
        prompt (str): The input prompt to guide the synthetic data generation.
        llm (ChatOpenAI): A ChatOpenAI language model instance used for generation.

    Returns:
        Questions: A structured output object containing the generated synthetic data.

    """
    structured_llm = llm.with_structured_output(Questions)
    response = structured_llm.invoke(prompt)
    return response

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def sgen(cfg: DictConfig) -> None:
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    if not (save_dir := Path(cfg.syntheticgen.save_dir)).exists():
        save_dir.mkdir(parents=True)

    for dept, details in cfg.syntheticgen.departments.items():
        logger.info(f"Generating synthetic data for department: {dept}")
        prompt = details.prompt
        questions = synthetic_data_generation(prompt, llm)
        # Store synthetic data as needed
        filename = details.filename
        with open(save_dir / filename, 'w') as f:
            json.dump(questions.model_dump(), f)


if __name__ == "__main__":
    load_dotenv()

    sgen()


    logger.info("Synthetic data and Chroma vector store created successfully.")