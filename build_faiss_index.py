import json
import os
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from logger import Logger


def find_json_files(directory: Path = None) -> list[Path]:
    """
    Find all JSON files in the specified directory (or current directory if None).
    """
    if directory is None:
        directory = Path(".")
    
    json_files = list(directory.glob("*.json"))
    return json_files


def get_logger() -> Logger:
    """
    Create a Logger instance in DEV mode,
    logging to Logs/faiss_build.log and console.
    """
    logs_dir = Path("Logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = logs_dir / "faiss_build.log"

    logger = Logger(
        name="FAISSBuilder",
        log_file_needed=True,
        log_file=str(log_file_path),
        level="DEV",
    )
    return logger

def get_embedding_model(logger: Logger):
    """
    Load HuggingFace sentence-transformers embedding model:
    sentence-transformers/all-MiniLM-L6-v2
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    logger.info(f"Loading embedding model: {model_name}")
    embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.debug("Embedding model loaded successfully.")
    return embedding

def load_sdg_json_as_documents(json_path: Path, logger: Logger) -> list[Document]:
    """
    Load sdg.json (list of SDG objects) and convert into LangChain Documents.
    """
    logger.info(f"Loading JSON from: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        sdgs = json.load(f)

    logger.debug(f"Found {len(sdgs)} entries in JSON.")
    docs: list[Document] = []

    for sdg in sdgs:
        sdg_id = sdg.get("sdg_id")
        sdg_name = sdg.get("sdg_name", f"SDG {sdg_id}")
        base_meta = {
            "sdg_id": sdg_id,
            "sdg_name": sdg_name,
        }
        headline = sdg.get("headline", "")
        summary = sdg.get("summary", "")
        text = f"{sdg_name} (SDG {sdg_id})\nHeadline: {headline}\nSummary: {summary}"
        docs.append(
            Document(
                page_content=text,
                metadata={**base_meta, "type": "summary"},
            )
        )
        for t in sdg.get("key_targets", []):
            docs.append(
                Document(
                    page_content=f"{sdg_name} - Target: {t}",
                    metadata={**base_meta, "type": "target"},
                )
            )
        for ex in sdg.get("business_examples", []):
            title = ex.get("title", "Business example")
            desc = ex.get("description", "")
            ex_text = f"{sdg_name} - Example: {title}\n{desc}"
            docs.append(
                Document(
                    page_content=ex_text,
                    metadata={**base_meta, "type": "business_example"},
                )
            )

    logger.debug(f"Converted SDG JSON to {len(docs)} document chunks.")
    return docs

def load_generic_json_as_documents(json_path: Path, logger: Logger) -> list[Document]:
    """
    Load any JSON file and convert into LangChain Documents.
    Handles various JSON structures by flattening content.
    """
    logger.info(f"Loading generic JSON from: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    docs: list[Document] = []
    filename = json_path.stem

    def extract_text_from_obj(obj, path=""):
        """Recursively extract text content from JSON objects"""
        texts = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if isinstance(value, (dict, list)):
                    texts.extend(extract_text_from_obj(value, current_path))
                elif isinstance(value, str) and len(value.strip()) > 10:
                    texts.append(f"{key}: {value.strip()}")
                elif value is not None:
                    texts.append(f"{key}: {str(value)}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                current_path = f"{path}[{i}]" if path else f"item_{i}"
                if isinstance(item, (dict, list)):
                    texts.extend(extract_text_from_obj(item, current_path))
                elif isinstance(item, str) and len(item.strip()) > 10:
                    texts.append(item.strip())
                elif item is not None:
                    texts.append(str(item))
        elif isinstance(obj, str) and len(obj.strip()) > 10:
            texts.append(obj.strip())
            
        return texts

    # Extract all text content
    all_texts = extract_text_from_obj(data)
    
    # Group texts into chunks to avoid too many small documents
    chunk_size = 5  # Number of text pieces per document
    
    for i in range(0, len(all_texts), chunk_size):
        chunk_texts = all_texts[i:i+chunk_size]
        content = "\n".join(chunk_texts)
        
        if content.strip():  # Only create document if there's actual content
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": str(json_path),
                        "filename": filename,
                        "chunk_id": i // chunk_size,
                        "type": "json_content",
                    },
                )
            )

    logger.debug(f"Converted generic JSON to {len(docs)} document chunks.")
    return docs

def is_sdg_format(json_path: Path) -> bool:
    """
    Check if the JSON file follows the SDG format (list of objects with sdg_id).
    """
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Check if it's a list and first item has sdg_id
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            return "sdg_id" in data[0]
        return False
    except:
        return False

def load_txt_as_documents(txt_path: Path, logger: Logger) -> list[Document]:
    """
    Load a plain .txt file and split into smaller chunks for RAG.
    """
    logger.debug(f"Loading text file from: {txt_path}")
    text = txt_path.read_text(encoding="utf-8")
    logger.debug(f"Raw text length: {len(text)} characters.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = splitter.split_text(text)
    logger.debug(f"Split text into {len(chunks)} chunks.")

    docs: list[Document] = []
    for i, chunk in enumerate(chunks):
        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": str(txt_path),
                    "chunk_id": i,
                    "type": "text_chunk",
                },
            )
        )

    return docs

def build_faiss_index(docs: list[Document], index_path: Path, logger: Logger):
    """
    Build and persist a FAISS index using all-MiniLM-L6-v2 embeddings.
    """
    if not docs:
        logger.error("No documents provided to build index.")
        raise ValueError("No documents provided to build index.")

    logger.info(f"Building FAISS index from {len(docs)} documents...")
    embedding = get_embedding_model(logger)

    vectorstore = FAISS.from_documents(docs, embedding)
    logger.debug("FAISS index created in memory.")

    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))

    logger.info(f"Saved FAISS index with {len(docs)} documents to {index_path}")

def main():
    logger = get_logger()
    logger.info("Starting FAISS index build process for all JSON files...")

    # Find all JSON files in current directory
    json_files = find_json_files()
    
    if not json_files:
        logger.error("No JSON files found in current directory.")
        return
    
    logger.info(f"Found {len(json_files)} JSON files to process: {[f.name for f in json_files]}")
    
    for json_file in json_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {json_file.name}")
        logger.info(f"{'='*60}")
        
        # Create index directory name from JSON filename (without .json extension)
        index_name = json_file.stem  # Gets filename without extension
        index_path = Path(index_name)
        
        logger.info(f"Input file: {json_file}")
        logger.info(f"Output index directory: {index_path}")
        
        try:
            # Detect JSON format and use appropriate loader
            if is_sdg_format(json_file):
                logger.info(f"Detected SDG format for {json_file.name}")
                docs = load_sdg_json_as_documents(json_file, logger)
            else:
                logger.info(f"Using generic JSON loader for {json_file.name}")
                docs = load_generic_json_as_documents(json_file, logger)
            
            # Build FAISS index
            build_faiss_index(docs, index_path, logger)
            logger.info(f"Successfully created vector store for {json_file.name}")
            
        except Exception as e:
            logger.error(f"Failed to process {json_file.name}: {str(e)}")
            continue
    
    logger.info("\nFAISS index build process completed for all files.")


if __name__ == "__main__":
    main()
