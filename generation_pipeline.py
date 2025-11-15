import os
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

import torch
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from logger import Logger
from causal_model_handler import ModelHandler


class MultiVectorStoreRAGPipeline:
    """
    Multi-Vector Store RAG pipeline:
    - loads multiple FAISS indices from different knowledge bases
    - retrieves relevant chunks from all knowledge sources for an idea
    - builds a comprehensive prompt
    - uses a local HF causal LM to generate a comprehensive analysis
    """

    def __init__(
        self,
        index_dirs: List[str] = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        quantize: bool = False,
        logs_dir: str = "Logs",
    ):
        if index_dirs is None:
            # Default to all available vector stores
            index_dirs = ["sdgs", "eu_taxonomy_user_guide", "csrd_fact_sheet", "un_2030_agenda"]
        
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, "generation.log")
        self.logger = Logger(
            name="MultiVectorStoreRAGPipeline",
            log_file_needed=True,
            log_file=log_file,
            level="DEV",
        )
        self.logger.info("Initializing Multi-Vector Store RAG pipeline...")

        self.logger.info(
            f"Loading embedding model for retrieval: {embedding_model_name}"
        )
        self.embedding = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Load multiple vector stores
        self.vectorstores = {}
        self.knowledge_base_names = {}
        
        for index_dir in index_dirs:
            index_path = Path(index_dir)
            if index_path.exists():
                self.logger.info(f"Loading FAISS index from: {index_path}")
                try:
                    vectorstore = FAISS.load_local(
                        str(index_path),
                        self.embedding,
                        allow_dangerous_deserialization=True,
                    )
                    self.vectorstores[index_dir] = vectorstore
                    # Create friendly names for knowledge bases
                    friendly_names = {
                        "sdgs": "UN Sustainable Development Goals",
                        "eu_taxonomy_user_guide": "EU Taxonomy", 
                        "csrd_fact_sheet": "Corporate Sustainability Reporting Directive",
                        "un_2030_agenda": "UN 2030 Agenda"
                    }
                    self.knowledge_base_names[index_dir] = friendly_names.get(index_dir, index_dir.replace("_", " ").title())
                    self.logger.info(f"Successfully loaded {self.knowledge_base_names[index_dir]} vectorstore")
                except Exception as e:
                    self.logger.warning(f"Failed to load vectorstore from {index_dir}: {e}")
            else:
                self.logger.warning(f"FAISS index directory not found: {index_path}")
        
        if not self.vectorstores:
            self.logger.error("No vector stores could be loaded!")
            raise FileNotFoundError("No valid FAISS index directories found")
        
        self.logger.info(f"Loaded {len(self.vectorstores)} vector stores: {list(self.vectorstores.keys())}")

        self.logger.info(f"Loading LLM: {llm_model_name}")
        self.model_handler = ModelHandler(model_name=llm_model_name, quantize=quantize)
        self.model, self.tokenizer = self.model_handler.load_model()
        self.model.eval()
        self.logger.info("LLM loaded and set to eval mode.")


    def retrieve_documents(self, idea: str, k_per_source: int = 6) -> Dict[str, List[Document]]:
        """Retrieve top-k documents from each FAISS index for a given idea."""
        self.logger.info(f"Retrieving top {k_per_source} chunks per knowledge base for idea.")
        all_docs = {}
        
        for source_name, vectorstore in self.vectorstores.items():
            try:
                docs = vectorstore.similarity_search(idea, k=k_per_source)
                all_docs[source_name] = docs
                self.logger.debug(f"Retrieved {len(docs)} documents from {self.knowledge_base_names[source_name]}")
            except Exception as e:
                self.logger.warning(f"Failed to retrieve from {source_name}: {e}")
                all_docs[source_name] = []
        
        total_docs = sum(len(docs) for docs in all_docs.values())
        self.logger.debug(f"Retrieved total of {total_docs} documents from {len(all_docs)} knowledge bases.")
        return all_docs

    @staticmethod
    def _group_docs_by_sdg(docs: List[Document]) -> Dict[Tuple[int, str], List[Document]]:
        grouped = defaultdict(list)
        for d in docs:
            sdg_id = d.metadata.get("sdg_id", -1)
            sdg_name = d.metadata.get("sdg_name", "Unknown SDG")
            grouped[(sdg_id, sdg_name)].append(d)
        return grouped

    def build_context(self, all_docs: Dict[str, List[Document]]) -> str:
        """
        Turn retrieved docs into a comprehensive context organized by knowledge source.
        """
        parts: List[str] = []
        
        for source_name, docs in all_docs.items():
            if not docs:
                continue
                
            knowledge_base_title = self.knowledge_base_names.get(source_name, source_name)
            parts.append(f"\n=== {knowledge_base_title.upper()} ===")
            
            # Check if this is SDG format (has sdg_id) or generic format
            sdg_docs = [d for d in docs if d.metadata.get("sdg_id") is not None]
            
            if sdg_docs:
                # Handle SDG format with grouping
                grouped = self._group_docs_by_sdg(sdg_docs)
                for (sdg_id, sdg_name), sdg_docs_group in grouped.items():
                    parts.append(f"\n--- SDG {sdg_id}: {sdg_name} ---")
                    for d in sdg_docs_group:
                        doc_type = d.metadata.get("type", "content")
                        parts.append(f"[{doc_type}] {d.page_content.strip()}")
            else:
                # Handle generic format
                for i, d in enumerate(docs):
                    doc_type = d.metadata.get("type", "content")
                    filename = d.metadata.get("filename", "")
                    parts.append(f"[{doc_type}] {d.page_content.strip()}")
            
            parts.append("")  # Add spacing between knowledge sources

        context = "\n".join(parts)
        self.logger.debug(f"Built comprehensive context of length {len(context)} characters.")
        return context

    def _clean_response(self, text: str) -> str:
        """
        Clean up the generated response by removing repetitive content and formatting properly.
        """
        lines = text.split('\n')
        cleaned_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if not line:
                if cleaned_lines and cleaned_lines[-1] != "":
                    cleaned_lines.append("")
                continue
                
            line_lower = line.lower()
            if line_lower not in seen_lines and len(line) < 500:
                cleaned_lines.append(line)
                seen_lines.add(line_lower)
        
        while cleaned_lines and cleaned_lines[-1] == "":
            cleaned_lines.pop()
            
        result = '\n'.join(cleaned_lines)
        
        if result.startswith("Analysis:"):
            result = result[9:].strip()
            
        return result


    def build_prompt(self, idea: str, context: str) -> str:
        """
        Build the instruction-style prompt for comprehensive multi-source analysis.
        """
        knowledge_sources = ", ".join(self.knowledge_base_names.values())
        
        prompt = f"""You are an expert sustainability advisor with deep knowledge of multiple regulatory frameworks and standards.

You will analyze a company idea against multiple authoritative sources: {knowledge_sources}.

Company idea:
{idea}

Reference information from multiple knowledge bases:
{context}

Please provide a comprehensive sustainability analysis that:

1. **Overall Assessment**: Brief summary of the idea's sustainability potential (2-3 sentences)

2. **Key Alignments**: Identify the most relevant standards, goals, or frameworks from the reference material. For each alignment:
   - State the specific standard/goal (e.g., SDG 7, EU Taxonomy criteria, etc.)
   - Give a relevance score (0-100)
   - Explain the connection in 2-3 sentences with specific references

3. **Regulatory Considerations**: Any compliance requirements or reporting obligations

4. **Recommendations**: 2-3 actionable suggestions for strengthening sustainability alignment

Format your response as clear, readable text with proper headings. Focus on the most significant alignments and practical insights.

Analysis:"""
        self.logger.debug(f"Prompt length: {len(prompt)} characters.")
        return prompt

    def generate(
        self,
        idea: str,
        k_per_source: int = 6,
        max_new_tokens: int = 768,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        """
        Full multi-source RAG pipeline:
        - retrieve docs from all knowledge bases
        - build comprehensive context
        - build prompt
        - run generation
        Returns cleaned, comprehensive analysis.
        """
        self.logger.info("Starting comprehensive multi-source RAG generation for new idea.")

        all_docs = self.retrieve_documents(idea, k_per_source=k_per_source)
        context = self.build_context(all_docs)
        prompt = self.build_prompt(idea, context)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        self.logger.debug(
            f"Tokenized input length: {inputs['input_ids'].shape[1]} tokens."
        )

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        
        cleaned_text = self._clean_response(text)

        self.logger.debug("Generation completed.")
        self.logger.debug(f"Generated text: {cleaned_text[:300]}...")
        return cleaned_text
