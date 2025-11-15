import streamlit as st
import os
import torch
from pathlib import Path
import time
from typing import List, Dict

from generation_pipeline import MultiVectorStoreRAGPipeline
from logger import Logger


# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline_loaded" not in st.session_state:
    st.session_state.pipeline_loaded = False


def get_logger() -> Logger:
    """
    Create a Logger instance for Streamlit app.
    """
    logs_dir = Path("Logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = logs_dir / "streamlit_app.log"

    logger = Logger(
        name="StreamlitApp",
        log_file_needed=True,
        log_file=str(log_file_path),
        level="DEV",
    )
    return logger


def adaptive_document_retrieval(pipeline, company_idea: str, relevance_threshold: float = 0.3):
    """
    Retrieve documents with adaptive filtering based on relevance scores.
    Only includes documents above the relevance threshold.
    """
    retrieved_docs = {}
    retrieval_stats = {}
    
    for source_name, vectorstore in pipeline.vectorstores.items():
        try:
            docs_with_scores = vectorstore.similarity_search_with_score(company_idea, k=10)
            
            if docs_with_scores:
                max_distance = max(score for _, score in docs_with_scores)
                min_distance = min(score for _, score in docs_with_scores)
                
                relevant_docs = []
                for doc, distance in docs_with_scores:
                    if max_distance > min_distance:
                        similarity = 1 - ((distance - min_distance) / (max_distance - min_distance))
                    else:
                        similarity = 1.0
                    
                    if similarity >= relevance_threshold:
                        relevant_docs.append(doc)
                
                retrieved_docs[source_name] = relevant_docs
                retrieval_stats[source_name] = {
                    'total_candidates': len(docs_with_scores),
                    'relevant_count': len(relevant_docs),
                    'threshold_used': relevance_threshold,
                    'avg_similarity': sum(1 - ((score - min_distance) / (max_distance - min_distance) if max_distance > min_distance else 0) for _, score in docs_with_scores[:len(relevant_docs)]) / len(relevant_docs) if relevant_docs else 0
                }
            else:
                retrieved_docs[source_name] = []
                retrieval_stats[source_name] = {'total_candidates': 0, 'relevant_count': 0, 'threshold_used': relevance_threshold, 'avg_similarity': 0}
                
        except Exception as e:
            retrieved_docs[source_name] = []
            retrieval_stats[source_name] = {'error': str(e)}
    
    return retrieved_docs, retrieval_stats


@st.cache_resource
def load_pipeline():
    """
    Load the Multi-Vector Store RAG pipeline with caching for better performance.
    """
    logger = get_logger()
    logger.info("Loading Multi-Vector Store RAG Pipeline for Streamlit...")
    
    try:
        pipeline = MultiVectorStoreRAGPipeline(
            llm_model_name="meta-llama/Llama-3.2-1B-Instruct",
            quantize=False,
        )
        logger.info("Pipeline loaded successfully")
        return pipeline, logger
    except Exception as e:
        logger.error(f"Failed to load pipeline: {str(e)}")
        st.error(f"Failed to load pipeline: {str(e)}")
        return None, logger


def display_knowledge_bases(pipeline, logger):
    """
    Display information about loaded knowledge bases in the sidebar.
    """
    with st.sidebar:
        st.subheader("Knowledge Bases")
        if hasattr(pipeline, 'vectorstores') and hasattr(pipeline, 'knowledge_base_names'):
            for source_name, friendly_name in pipeline.knowledge_base_names.items():
                if source_name in pipeline.vectorstores:
                    st.success(f"✓ {friendly_name}")
                    logger.debug(f"Knowledge base displayed: {friendly_name}")
        else:
            st.warning("No knowledge bases loaded")
            
        st.subheader("About")
        st.write("""
        This app analyzes company ideas against multiple sustainability frameworks:
        - **UN SDGs**: Sustainable Development Goals
        - **EU Taxonomy**: Environmental objectives
        - **CSRD**: Corporate reporting requirements  
        - **UN 2030 Agenda**: Global sustainability targets
        
        Each analysis shows the specific sources and documents used to ensure transparency and accountability.
        """)
        
        # Clear chat button
        st.markdown("---")
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            logger.info("Chat history cleared by user")
            st.rerun()


def format_source_info(doc: object, index: int) -> tuple:
    """
    Extract and format source information from a document.
    """
    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
    
    # Try to get meaningful identifiers
    source_id = metadata.get('source', metadata.get('id', f'Document {index}'))
    title = metadata.get('title', metadata.get('name', ''))
    
    # Create a meaningful display title
    if title:
        display_title = f"{index}. {title}"
    else:
        display_title = f"{index}. {source_id}"
    
    # Create content preview
    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
    content_preview = content[:300] + "..." if len(content) > 300 else content
    
    return display_title, content_preview, metadata


def display_sources_with_stats(retrieved_docs: Dict[str, List], retrieval_stats: Dict, pipeline, logger):
    """
    Display the sources used in analysis with retrieval statistics for better explainability.
    """
    st.markdown("---")
    st.markdown("### Sources and Explainability")
    
    # Show retrieval methodology first
    with st.expander("How Sources Were Selected"):
        st.markdown("""
        **Adaptive Retrieval Process:**
        1. Search each knowledge base for relevant documents
        2. Calculate relevance scores based on semantic similarity
        3. Apply relevance threshold (30%) to filter out less relevant sources
        4. Only include documents that meet the relevance criteria
        
        This ensures that only truly relevant sources inform the analysis.
        """)
    
    # Group sources by knowledge base with statistics
    sources_by_kb = {}
    total_sources = 0
    total_candidates = 0
    
    for source_name, docs in retrieved_docs.items():
        kb_friendly_name = pipeline.knowledge_base_names.get(source_name, source_name)
        stats = retrieval_stats.get(source_name, {})
        
        if docs:  # Only include sources that have documents
            sources_by_kb[kb_friendly_name] = {'docs': docs, 'stats': stats}
            total_sources += len(docs)
        
        total_candidates += stats.get('total_candidates', 0)
    
    # Show overall retrieval statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sources Found", f"{total_sources}")
    with col2:
        st.metric("Knowledge Bases", f"{len(sources_by_kb)}")
    with col3:
        st.metric("Total Searched", f"{total_candidates}")
    with col4:
        relevance_rate = (total_sources / total_candidates * 100) if total_candidates > 0 else 0
        st.metric("Relevance Rate", f"{relevance_rate:.1f}%")
    
    if sources_by_kb:
        st.markdown("**Sources that informed this analysis:**")
        
        # Create tabs for each knowledge base with enhanced info
        kb_names = list(sources_by_kb.keys())
        if len(kb_names) > 1:
            tab_labels = []
            for name in kb_names:
                docs_count = len(sources_by_kb[name]['docs'])
                stats = sources_by_kb[name]['stats']
                avg_sim = stats.get('avg_similarity', 0)
                tab_labels.append(f"{name} ({docs_count} docs, {avg_sim:.0%} avg relevance)")
            
            tabs = st.tabs(tab_labels)
            
            for tab, kb_name in zip(tabs, kb_names):
                with tab:
                    docs = sources_by_kb[kb_name]['docs']
                    stats = sources_by_kb[kb_name]['stats']
                    
                    # Show knowledge base statistics
                    st.markdown(f"**Retrieval Statistics for {kb_name}:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.text(f"Relevant: {stats.get('relevant_count', 0)}")
                    with col2:
                        st.text(f"Searched: {stats.get('total_candidates', 0)}")
                    with col3:
                        st.text(f"Avg Relevance: {stats.get('avg_similarity', 0):.0%}")
                    
                    st.markdown("---")
                    
                    # Show documents
                    for i, doc in enumerate(docs[:5], 1):  # Show top 5 sources per KB
                        display_title, content_preview, metadata = format_source_info(doc, i)
                        
                        with st.expander(display_title):
                            st.markdown(f"**Content Preview:**")
                            st.markdown(f"*{content_preview}*")
                            
                            if metadata:
                                st.markdown("**Metadata:**")
                                for key, value in metadata.items():
                                    if key not in ['source', 'id', 'title', 'name']:  # Don't repeat already shown info
                                        st.text(f"{key}: {value}")
                            st.markdown(f"**Content Preview:**")
                            st.markdown(f"*{content_preview}*")
                            
                            if metadata:
                                st.markdown("**Metadata:**")
                                for key, value in metadata.items():
                                    if key not in ['source', 'id', 'title', 'name']:  # Don't repeat already shown info
                                        st.text(f"{key}: {value}")
        else:
            # Single knowledge base - show directly
            kb_name = kb_names[0]
            kb_data = sources_by_kb[kb_name]
            docs = kb_data['docs']
            stats = kb_data['stats']
            
            # Show statistics
            st.markdown(f"**Retrieval Statistics:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.text(f"Relevant: {stats.get('relevant_count', 0)}")
            with col2:
                st.text(f"Searched: {stats.get('total_candidates', 0)}")
            with col3:
                st.text(f"Avg Relevance: {stats.get('avg_similarity', 0):.0%}")
            
            for i, doc in enumerate(docs[:5], 1):  # Show top 5 sources
                display_title, content_preview, metadata = format_source_info(doc, i)
                
                with st.expander(display_title):
                    st.markdown(f"**Content Preview:**")
                    st.markdown(f"*{content_preview}*")
                    
                    if metadata:
                        st.markdown("**Metadata:**")
                        for key, value in metadata.items():
                            if key not in ['source', 'id', 'title', 'name']:
                                st.text(f"{key}: {value}")
                            if key not in ['source', 'id', 'title', 'name']:
                                st.text(f"{key}: {value}")
        
        # Enhanced summary with transparency metrics
        st.info(f"**Analysis Transparency**: {total_sources} relevant documents from {len(sources_by_kb)} knowledge bases ({relevance_rate:.1f}% relevance rate)")
        
    else:
        st.warning("No relevant sources were found for this analysis. Try rephrasing your idea or being more specific.")
    
    logger.info(f"Displayed {total_sources} relevant sources from {total_candidates} candidates across {len(sources_by_kb)} knowledge bases")


def main():
    # Configure page
    st.set_page_config(
        page_title="Sustainability Analysis Tool",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize logger
    logger = get_logger()
    logger.info("Streamlit app started")
    
    # Header
    st.title("Sustainability Analysis Tool")
    st.subheader("Analyze your company ideas against multiple sustainability frameworks")
    
    # Load pipeline
    with st.spinner("Loading AI models and knowledge bases..."):
        pipeline, logger = load_pipeline()
    
    if pipeline is None:
        st.error("Failed to load the analysis pipeline. Please check the logs.")
        logger.error("Pipeline loading failed, stopping app")
        return
    
    # Display knowledge bases in sidebar
    display_knowledge_bases(pipeline, logger)
    
    # Main interface
    st.markdown("---")
    
    # Chat interface
    st.subheader("Chat with the Sustainability Advisor")
    
    # Example ideas in expander
    with st.expander("Need inspiration? Click here for example ideas"):
        example_buttons = st.columns(2)
        
        with example_buttons[0]:
            if st.button("Solar + AI Manufacturing", use_container_width=True):
                st.session_state.user_input = "Solar panel manufacturing company with AI-powered efficiency optimization"
            if st.button("Vertical Farming", use_container_width=True):
                st.session_state.user_input = "Vertical farming startup using renewable energy and water recycling systems"
            if st.button("EV Charging Network", use_container_width=True):
                st.session_state.user_input = "Electric vehicle charging network powered entirely by renewable energy"
            if st.button("Sustainable Packaging", use_container_width=True):
                st.session_state.user_input = "Sustainable packaging company using biodegradable materials from agricultural waste"
                
        with example_buttons[1]:
            if st.button("Carbon Capture Tech", use_container_width=True):
                st.session_state.user_input = "Carbon capture technology for reducing industrial emissions"
            if st.button("Circular Economy Platform", use_container_width=True):
                st.session_state.user_input = "Digital platform connecting businesses for waste material exchange and reuse"
            if st.button("Ocean Plastic Buildings", use_container_width=True):
                st.session_state.user_input = "Green building materials manufactured from recycled ocean plastic"
            if st.button("Smart Grid Technology", use_container_width=True):
                st.session_state.user_input = "AI-powered smart grid technology for optimizing renewable energy distribution"
    
    # Chat messages container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(f"**Company Idea:** {message['content']}")
                else:
                    st.markdown(message["content"])
    
    # Chat input
    user_input = st.chat_input(
        "Describe your company idea and press Enter...",
        key="chat_input"
    )
    
    # Handle user input from chat or example buttons
    if user_input or ("user_input" in st.session_state and st.session_state.user_input):
        # Use example input if available, otherwise use chat input
        if "user_input" in st.session_state and st.session_state.user_input:
            company_idea = st.session_state.user_input
            st.session_state.user_input = ""  # Clear after use
        else:
            company_idea = user_input
            
        if company_idea and company_idea.strip():
            logger.info(f"New user message: {company_idea[:100]}...")
            
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": company_idea})
            
            # Display user message immediately
            with st.chat_message("user"):
                st.write(f"**Company Idea:** {company_idea}")
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your idea against sustainability frameworks..."):
                    try:
                        start_time = time.time()
                        
                        # Use adaptive retrieval for better explainability
                        retrieved_docs, retrieval_stats = adaptive_document_retrieval(pipeline, company_idea, relevance_threshold=0.3)
                        
                        # Create a custom context using our adaptive retrieval
                        context = pipeline.build_context(retrieved_docs)
                        prompt = pipeline.build_prompt(company_idea, context)
                        
                        # Generate analysis using the custom prompt
                        inputs = pipeline.tokenizer(
                            prompt,
                            return_tensors="pt",
                            truncation=True,
                            max_length=2048,
                        )
                        
                        device = next(pipeline.model.parameters()).device
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            output = pipeline.model.generate(
                                **inputs,
                                max_new_tokens=768,
                                do_sample=True,
                                temperature=0.2,
                                top_p=0.9,
                                pad_token_id=pipeline.tokenizer.eos_token_id,
                            )
                        
                        generated_ids = output[0][inputs["input_ids"].shape[1] :]
                        analysis = pipeline.tokenizer.decode(
                            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                        )
                        
                        # Clean the response
                        analysis = pipeline._clean_response(analysis)
                        
                        end_time = time.time()
                        analysis_time = end_time - start_time
                        
                        logger.info(f"Analysis completed in {analysis_time:.2f} seconds")
                        
                        # Display the main analysis
                        st.markdown("## Sustainability Analysis")
                        st.markdown(analysis)
                        
                        # Display sources for explainability with retrieval stats
                        display_sources_with_stats(retrieved_docs, retrieval_stats, pipeline, logger)
                        
                        # Add transparency metrics
                        st.markdown("---")
                        
                        total_docs = sum(len(docs) for docs in retrieved_docs.values())
                        active_kbs = len([kb for kb, docs in retrieved_docs.items() if docs])
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Analysis Time", f"{analysis_time:.1f}s")
                        with col2:
                            st.metric("Knowledge Bases", f"{active_kbs}")
                        with col3:
                            st.metric("Documents Used", f"{total_docs}")
                        with col4:
                            st.metric("Model", "Llama-3.2-1B")
                        
                        # Create comprehensive response for chat history
                        sources_summary = ""
                        if retrieved_docs:
                            kb_counts = {pipeline.knowledge_base_names.get(kb, kb): len(docs) 
                                       for kb, docs in retrieved_docs.items() if docs}
                            sources_list = [f"{kb} ({count})" for kb, count in kb_counts.items()]
                            sources_summary = f"\n\n**Sources Used:** {', '.join(sources_list)}"
                        
                        response = f"## Sustainability Analysis\n\n{analysis}{sources_summary}\n\n---\n*Analysis completed in {analysis_time:.1f} seconds using {total_docs} source documents*"
                        
                        # Create detailed report for download
                        st.markdown("---")
                        detailed_report = create_detailed_report(company_idea, analysis, retrieved_docs, pipeline, analysis_time)
                        
                        st.download_button(
                            label="Download Detailed Report with Sources",
                            data=detailed_report,
                            file_name=f"sustainability_analysis_{int(time.time())}.md",
                            mime="text/markdown",
                            key=f"download_{len(st.session_state.messages)}",
                            use_container_width=True
                        )
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        error_msg = f"Analysis failed: {str(e)}"
                        st.error(error_msg)
                        logger.error(f"Analysis failed: {str(e)}")
                        
                        # Add error to chat history
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            # Auto-scroll to bottom (rerun to show new messages)
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Built with Streamlit • Powered by Multi-Vector Store RAG • AI Sustainability Advisor</p>
        <p><small>Chat interface for analyzing company ideas against UN SDGs, EU Taxonomy, CSRD, and UN 2030 Agenda</small></p>
    </div>
    """, unsafe_allow_html=True)


def create_detailed_report(company_idea: str, analysis: str, retrieved_docs: Dict, pipeline, analysis_time: float) -> str:
    """
    Create a detailed report with sources for download.
    """
    report = f"""# Sustainability Analysis Report

## Company Idea
{company_idea}

## Analysis
{analysis}

## Detailed Sources and References

This analysis was generated using {sum(len(docs) for docs in retrieved_docs.values())} source documents from {len([kb for kb, docs in retrieved_docs.items() if docs])} knowledge bases.

"""
    
    for source_name, docs in retrieved_docs.items():
        if docs:
            kb_friendly_name = pipeline.knowledge_base_names.get(source_name, source_name)
            report += f"\n### {kb_friendly_name} ({len(docs)} documents)\n"
            
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                source_id = metadata.get('source', metadata.get('id', f'Document {i}'))
                title = metadata.get('title', metadata.get('name', source_id))
                
                report += f"\n#### {i}. {title}\n"
                
                # Add metadata if available
                if metadata:
                    report += "**Metadata:**\n"
                    for key, value in metadata.items():
                        report += f"- {key}: {value}\n"
                        report += f"- {key}: {value}\n"
                
                # Add content
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                report += f"\n**Content:**\n{content}\n\n---\n"
    
    report += f"""
## Analysis Metadata
- **Generation Time**: {analysis_time:.2f} seconds
- **Model Used**: Llama-3.2-1B-Instruct
- **Total Documents Retrieved**: {sum(len(docs) for docs in retrieved_docs.values())}
- **Knowledge Bases Used**: {len([kb for kb, docs in retrieved_docs.items() if docs])}
- **Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}

---
*This report was generated with full source attribution to ensure transparency and accountability in the analysis process.*
"""
    
    return report


if __name__ == "__main__":
    main()