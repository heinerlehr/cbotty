
import json
from pathlib import Path

# LangChain/LangGraph
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Configuration management
from utils.Configuration import Configuration

# Logging
from loguru import logger

from synthetic.sgen import Questions

#######################################################################################################################
#
# Specific task implementations
#
#######################################################################################################################

def cleandb(config: Configuration) -> None:
    """
    Delete a ChromaDB collection from the specified directory.

    This function creates a connection to a ChromaDB collection using the provided
    configuration and permanently deletes the entire collection from the persist
    directory.

    Args:
        cfg (DictConfig): Configuration object containing ChromaDB settings.
                         Must include:
                         - cfg.chroma.collection_name: Name of the collection to delete
                         - cfg.chroma.persist_directory: Directory where the collection is stored

    Returns:
        None

    """
    chroma_db = Chroma(
        collection_name=config('chroma', 'collection_name'),
        persist_directory=config('chroma', 'persist_directory')
    )
    chroma_db.delete_collection()
    logger.info(f"Deleted Chroma collection '{config('chroma', 'collection_name')}' from '{config('chroma', 'persist_directory')}'.")

def initdb(config: Configuration) -> None:
    """
    Initialize the database by loading synthetic data from configured departments.
    
    Iterates through all departments defined in the configuration's syntheticgen section
    and loads their corresponding data files into the ChromaDB collection.
    
    Args:
        config (Configuration): Configuration object
    Returns:
        None
    
    Raises:
        FileNotFoundError: If a department's data file doesn't exist
        ConfigurationError: If required configuration keys are missing
    """

    for dept, details in config('syntheticgen', 'departments').items():
        logger.info(f"Incorporating synthetic data for department: {dept}")
        # Store synthetic data as needed
        filename = details['filename']
        load_db_from_file(file_path=Path(config('syntheticgen', 'save_dir')) / filename, 
                          collection_name=config('chroma', 'collection_name'), 
                          persist_directory=config('chroma', 'persist_directory'))

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

def querydb(config: Configuration, query: str, nresults: int) -> list[Document]:
    """
    Query the database using a retriever and log the top results.
    
    Args:
        config (Configuration): Configuration object containing retriever settings
        query (str): The search query string to execute
        nresults (int): Number of top results to retrieve and display
        
    Returns:
        None: This function logs results but doesn't return any value
        
    Note:
        Results are logged with their content and metadata using the logger
    """

    retriever = get_retriever(config)
    return retriever.invoke(query, k=nresults)


def get_retriever(config: Configuration):
    """
    Create and configure a Chroma database retriever with OpenAI embeddings.
    
    This function initializes a Chroma vector database with OpenAI embeddings and
    returns a configured retriever for similarity search operations.
    
    Args:
        cfg (Configuration): Configuration object containing Chroma database settings.
    
    Returns:
        VectorStoreRetriever: Configured Chroma retriever instance for document search

    """
    embeddings = OpenAIEmbeddings()

    chroma_db = Chroma(
        collection_name=config('chroma', 'collection_name'),
        embedding_function=embeddings,
        persist_directory=config('chroma', 'persist_directory')
    )

    search_type=config('chroma').get('search_type', 'similarity_score_threshold')
    score_threshold=config('chroma', 'score_threshold')
    k=config('chroma', 'k')
    retriever = chroma_db.as_retriever(search_type=search_type, 
                                       search_kwargs={
                                           "k": k,
                                           "score_threshold": score_threshold
                                       })
    return retriever

#######################################################################################################################
#
# Infrastructure code for command line argument parsing and task launching
#
#######################################################################################################################


def main(config: Configuration) -> None:
    """
    Execute database operations based on the specified task configuration.

    This function serves as the main entry point for database operations, supporting
    multiple tasks including initialization, cleaning, and querying of the database.

    Args:
        config (Configuration): Configuration object containing task parameters.

    Returns:
        Any: Return value from the executed task function, or None if task
             is not recognized or an exception occurs.

    Raises:
        Exception: Logs any exceptions that occur during task execution without
                  re-raising them.

    Note:
        Supported tasks:
        - 'clean': Executes cleandb function
        - 'init': Executes initdb function  
        - 'query': Executes querydb function with query and nresults parameters
    """

    task = config.get('task', 'init')  # Default to 'init'
    ret = True
    kwargs = {}
    try:
        match task:
            case 'clean':
                func = cleandb
            case 'init':
                func = initdb
            case 'query':
                kwargs = {'query': config.get('query', ''), 'nresults': config.get('nresults', 5)}
                func = querydb
            case _:
                logger.error(f'Task {task} not recognized')
                ret = False
        if ret:
            return func(config=config, **kwargs)
    except Exception as e:
        logger.exception(f'An error occurred while executing task {task}: {e}')

# Parse arguments before Hydra processes them
parsed_args = None

if __name__ == "__main__":
    load_dotenv()
    config = Configuration()

    main(config)
