"""
Refactored generation pipeline with cleaner architecture.
Handles prompt engineering, context formatting, and response generation.
"""
from pathlib import Path
from typing import Dict, List, Optional
import torch
import random
import numpy as np

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from model_handler import ModelHandler
from retriever import SustainabilityRetriever


# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class SustainabilityGenerator:
    """Handles LLM-based generation for sustainability queries."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        quantize: bool = False,
        max_new_tokens: int = 1500
    ):
        """
        Initialize generator with model and retriever.
        
        Args:
            model_name: HuggingFace model ID
            quantize: Whether to use quantization
            max_new_tokens: Maximum tokens to generate
        """
        print(f"Loading model: {model_name}")
        handler = ModelHandler(model_name=model_name, quantize=quantize)
        base_model, tokenizer = handler.load_model()
        
        self.retriever = SustainabilityRetriever()
        
        # Create HuggingFace pipeline
        self.pipe = pipeline(
            "text-generation",
            model=base_model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for faithfulness
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False,
        )
        
        # LangChain wrapper
        self.lc_pipe = HuggingFacePipeline(pipeline=self.pipe)
        
        # Prompt template
        self.prompt = self._create_prompt()
        
        # Chain
        self.chain = self.prompt | self.lc_pipe | StrOutputParser()
    
    def _create_prompt(self) -> PromptTemplate:
        """Create structured prompt template for sustainability evaluation."""
        template = """You will produce a concise, fully-cited evaluation of the BUSINESS CONCEPT using ONLY the CONTEXT provided. Do NOT introduce outside knowledge.

INSTRUCTIONS:
- Use the context verbatim where possible and indicate source framework per item (SDGs, UN 2030 Agenda, EU Taxonomy, CSRD).
- For every alignment claim, include a short quotation from the context in quotation marks and the framework tag in square brackets. Example:
  SDG Mapping: SDG 7 "Ensure access to affordable, reliable, sustainable and modern energy for all" [SDGs]
- If a specific policy, target or ESRS code is not present in the context, write: "Not mentioned in the context." for that line.
- Keep answers factual and short (2-4 bullets per section). Do not speculate.

CONTEXT:
{context}

BUSINESS CONCEPT: {query}

OUTPUT FORMAT (strict):
1) SDG Mapping:
 - <SDG number and short name>: "<exact quote or paraphrase from context>" [SDGs]
 - ...

2) Policy & Standards Alignment:
 - <Policy/Standard name>: "<exact quote or paraphrase from context>" [Framework]
 - If not in context: "Not mentioned in the context."

3) EU Taxonomy Eligibility (if present):
 - "<quote or clause from context>" [EU Taxonomy]
 - If not in context: "Not mentioned in the context."

4) CSRD / ESRS relevance:
 - "<quote or clause from context>" [CSRD]
 - If not in context: "Not mentioned in the context."

5) Key gaps (explicit):
 - List short bullets of missing info that prevents definitive alignment statements (max 5 items).

Remember: Only use text present in the context. Where you quote, use quotation marks and append the framework tag in square brackets.
RESPONSE:"""
        return PromptTemplate.from_template(template)
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into structured context."""
        context_by_framework = {}
        for chunk in chunks[:25]:  # Max 25 chunks
            source = chunk['source']
            if source not in context_by_framework:
                context_by_framework[source] = []
            context_by_framework[source].append(chunk['text'])
        
        framework_names = {
            'un_2030_agenda': 'UN 2030 Agenda',
            'sdgs': 'Sustainable Development Goals (SDGs)',
            'eu_taxonomy': 'EU Taxonomy',
            'csrd': 'Corporate Sustainability Reporting Directive (CSRD)'
        }
        
        context_text = ""
        for framework_name, texts in context_by_framework.items():
            display_name = framework_names.get(framework_name, framework_name)
            context_text += f"\n=== {display_name.upper()} ===\n"
            context_text += "\n\n".join(texts[:6])  # Max 6 per framework
            context_text += "\n"
        
        return context_text
    
    def _trim_response(self, response: str) -> str:
        """Trim excessive key gaps and remove unnecessary sections."""
        # Limit key gaps to max 5 items
        if "5) Key gaps" in response:
            parts = response.split("5) Key gaps")
            if len(parts) == 2:
                head, tail = parts
                lines = tail.split('\n')
                gap_items = [ln for ln in lines if ln.strip().startswith('-')][:5]
                response = head + "5) Key gaps" + '\n' + '\n'.join(gap_items)
        
        return response.strip()
    
    def generate(
        self,
        query: str,
        top_k: int = 25,
        framework: Optional[str] = None
    ) -> Dict:
        """
        Generate sustainability evaluation for a business query.
        
        Args:
            query: Business concept description
            top_k: Number of context chunks to retrieve
            framework: Specific framework to focus on (None = all)
        
        Returns:
            Dict with 'answer', 'sources', 'retrieved_chunks'
        """
        # Retrieve relevant context
        retrieved_chunks = self.retriever.retrieve(
            query,
            top_k=top_k,
            framework=framework
        )
        
        if not retrieved_chunks:
            return {
                "answer": "No relevant information found in the sustainability frameworks.",
                "sources": [],
                "retrieved_chunks": []
            }
        
        # Format context
        context = self._format_context(retrieved_chunks)
        
        # Generate response
        response = self.chain.invoke({"context": context, "query": query})
        response = self._trim_response(response)
        
        # Extract sources
        sources = {}
        for chunk in retrieved_chunks:
            source = chunk['source']
            sources[source] = sources.get(source, 0) + 1
        
        framework_names = {
            'un_2030_agenda': 'UN 2030 Agenda for Sustainable Development',
            'sdgs': 'Sustainable Development Goals (SDGs)',
            'eu_taxonomy': 'EU Taxonomy for Sustainable Activities',
            'csrd': 'Corporate Sustainability Reporting Directive (CSRD)'
        }
        
        source_list = [
            f"{framework_names.get(src, src)} ({count} sections)"
            for src, count in sources.items()
        ]
        
        return {
            "answer": response,
            "sources": source_list,
            "retrieved_chunks": retrieved_chunks
        }


# Global instance for backward compatibility
_generator = None

def get_generator() -> SustainabilityGenerator:
    """Get or create global generator instance."""
    global _generator
    if _generator is None:
        _generator = SustainabilityGenerator()
    return _generator


def answer_sustainability_query(query: str, framework: Optional[str] = None) -> str:
    """
    Simple function interface for backward compatibility.
    
    Args:
        query: Business concept to evaluate
        framework: Optional framework to focus on
    
    Returns:
        Generated evaluation text
    """
    generator = get_generator()
    result = generator.generate(query, framework=framework)
    return result["answer"]


if __name__ == "__main__":
    # Test generation
    generator = SustainabilityGenerator()
    
    test_query = "Our startup provides solar panel leasing for small businesses in rural areas."
    print(f"\nQuery: {test_query}\n")
    
    result = generator.generate(test_query)
    
    print("Response:")
    print(result["answer"])
    print(f"\nSources: {', '.join(result['sources'])}")
