
import json
from pathlib import Path

# LangChain/LangGraph
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Configuration management
import hydra
from omegaconf import DictConfig

# Logging
from loguru import logger

from synthetic.sgen import Questions

#######################################################################################################################
#
# Specific task implementations
#
#######################################################################################################################

def cleandb(cfg: DictConfig) -> None:
    chroma_db = Chroma(
        collection_name=cfg.chroma.collection_name,
        persist_directory=cfg.chroma.persist_directory
    )
    chroma_db.delete_collection()
    logger.info(f"Deleted Chroma collection '{cfg.chroma.collection_name}' from '{cfg.chroma.persist_directory}'.")

def initdb(cfg: DictConfig) -> None:

    for dept, details in cfg.syntheticgen.departments.items():
        logger.info(f"Incorporating synthetic data for department: {dept}")
        # Store synthetic data as needed
        filename = details.filename
        load_db_from_file(file_path=Path(cfg.syntheticgen.save_dir) / filename, 
                          collection_name=cfg.chroma.collection_name, 
                          persist_directory=cfg.chroma.persist_directory)
    

def load_db_from_file(file_path: Path, collection_name: str, persist_directory: str) -> None:
    """
    Load synthetic data from a JSON file and populate a Chroma vector store.

    Args:
        file_path (Path): Path to the JSON file containing synthetic data.
        collection_name (str): Name of the Chroma collection to create or update.
        persist_directory (str): Directory where the Chroma database is persisted.

    Returns:
        None
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    questions = Questions.model_validate(data)

    documents = []
    for q in questions.questions:
        doc = Document(
            page_content=q.question + "\n" + q.answer,
            metadata={"category": q.category}
        )
        documents.append(doc)

    embeddings = OpenAIEmbeddings()
    chroma_db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    chroma_db.add_documents(documents)
    logger.info(f"Loaded {len(documents)} documents into Chroma collection '{collection_name}'.")

def querydb(cfg: DictConfig, query: str, nresults: int) -> None:

    retriever = get_retriever(cfg)
    results = retriever.invoke(query, k=nresults)
    logger.info(f"Top {nresults} results for query '{query}':")
    for i, doc in enumerate(results, start=1):
        logger.info(f"Result {i}: {doc.page_content} | Metadata: {doc.metadata}")

def get_retriever(cfg: DictConfig):
    embeddings = OpenAIEmbeddings()

    chroma_db = Chroma(
        collection_name=cfg.chroma.collection_name,
        embedding_function=embeddings,
        persist_directory=cfg.chroma.persist_directory
    )
    retriever = chroma_db.as_retriever(search_type=cfg.chroma.get('search_type', 'similarity_score_threshold'),
                                       search_kwargs={"k": cfg.chroma.k, "score_threshold": cfg.chroma.score_threshold})
    return retriever

#######################################################################################################################
#
# Infrastructure code for command line argument parsing and task launching
#
#######################################################################################################################


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:

    task = cfg.get('task', 'init')  # Default to 'init'
    ret = True
    kwargs = {}
    try:
        match task:
            case 'clean':
                func = cleandb
            case 'init':
                func = initdb
            case 'query':
                kwargs = {'query': cfg.get('query', ''), 'nresults': cfg.get('nresults', 5)}
                func = querydb
            case _:
                logger.error(f'Task {task} not recognized')
                ret = False
        if ret:
            return func(cfg=cfg, **kwargs)
    except Exception as e:
        logger.exception(f'An error occurred while executing task {task}: {e}')

# Parse arguments before Hydra processes them
parsed_args = None

if __name__ == "__main__":
    load_dotenv()
    main()
