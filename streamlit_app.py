import streamlit as st
import torch
from pathlib import Path
import time
from typing import List, Dict
from io import BytesIO

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import black, blue, grey
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from generation_pipeline import MultiVectorStoreRAGPipeline
from logger import Logger


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline_loaded" not in st.session_state:
    st.session_state.pipeline_loaded = False
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None


def get_logger() -> Logger:
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
        
        st.markdown("---")
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            logger.info("Chat history cleared by user")
            st.rerun()


def format_source_info(doc: object, index: int) -> tuple:
    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
    source_id = metadata.get('source', metadata.get('id', f'Document {index}'))
    title = metadata.get('title', metadata.get('name', ''))
    
    if title:
        display_title = f"{index}. {title}"
    else:
        display_title = f"{index}. {source_id}"
    
    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
    content_preview = content[:300] + "..." if len(content) > 300 else content
    
    return display_title, content_preview, metadata


def display_sources_with_stats(retrieved_docs: Dict[str, List], retrieval_stats: Dict, pipeline, logger):
    st.markdown("---")
    st.markdown("### Sources and Explainability")
    
    with st.expander("How Sources Were Selected"):
        st.markdown("""
        **Adaptive Retrieval Process:**
        1. Search each knowledge base for relevant documents
        2. Calculate relevance scores based on semantic similarity
        3. Apply relevance threshold (30%) to filter out less relevant sources
        4. Only include documents that meet the relevance criteria
        
        This ensures that only truly relevant sources inform the analysis.
        """)
    
    sources_by_kb = {}
    total_sources = 0
    total_candidates = 0
    
    for source_name, docs in retrieved_docs.items():
        kb_friendly_name = pipeline.knowledge_base_names.get(source_name, source_name)
        stats = retrieval_stats.get(source_name, {})
        
        if docs:
            sources_by_kb[kb_friendly_name] = {'docs': docs, 'stats': stats}
            total_sources += len(docs)
        
        total_candidates += stats.get('total_candidates', 0)
    
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
                    
                    st.markdown(f"**Retrieval Statistics for {kb_name}:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.text(f"Relevant: {stats.get('relevant_count', 0)}")
                    with col2:
                        st.text(f"Searched: {stats.get('total_candidates', 0)}")
                    with col3:
                        st.text(f"Avg Relevance: {stats.get('avg_similarity', 0):.0%}")
                    
                    st.markdown("---")
                    
                    for i, doc in enumerate(docs[:5], 1):
                        display_title, content_preview, metadata = format_source_info(doc, i)
                        
                        with st.expander(display_title):
                            st.markdown(f"**Content Preview:**")
                            st.markdown(f"*{content_preview}*")
                            
                            if metadata:
                                st.markdown("**Metadata:**")
                                for key, value in metadata.items():
                                    if key not in ['source', 'id', 'title', 'name']:
                                        st.text(f"{key}: {value}")
        else:
            kb_name = kb_names[0]
            kb_data = sources_by_kb[kb_name]
            docs = kb_data['docs']
            stats = kb_data['stats']
            
            st.markdown(f"**Retrieval Statistics:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.text(f"Relevant: {stats.get('relevant_count', 0)}")
            with col2:
                st.text(f"Searched: {stats.get('total_candidates', 0)}")
            with col3:
                st.text(f"Avg Relevance: {stats.get('avg_similarity', 0):.0%}")
            
            for i, doc in enumerate(docs[:5], 1):
                display_title, content_preview, metadata = format_source_info(doc, i)
                
                with st.expander(display_title):
                    st.markdown(f"**Content Preview:**")
                    st.markdown(f"*{content_preview}*")
                    
                    if metadata:
                        st.markdown("**Metadata:**")
                        for key, value in metadata.items():
                            if key not in ['source', 'id', 'title', 'name']:
                                st.text(f"{key}: {value}")
        
        st.info(f"**Analysis Transparency**: {total_sources} relevant documents from {len(sources_by_kb)} knowledge bases ({relevance_rate:.1f}% relevance rate)")
        
    else:
        st.warning("No relevant sources were found for this analysis. Try rephrasing your idea or being more specific.")
    
    logger.info(f"Displayed {total_sources} relevant sources from {total_candidates} candidates across {len(sources_by_kb)} knowledge bases")


def create_detailed_report(company_idea: str, analysis: str, retrieved_docs: Dict, pipeline, analysis_time: float) -> str:
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
                
                if metadata:
                    report += "**Metadata:**\n"
                    for key, value in metadata.items():
                        report += f"- {key}: {value}\n"
                
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


def create_pdf_report(company_idea: str, analysis: str, retrieved_docs: Dict, pipeline, analysis_time: float) -> BytesIO:
    if not REPORTLAB_AVAILABLE:
        raise ImportError("ReportLab is required for PDF generation. Install it with: pip install reportlab")
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch, bottomMargin=1*inch)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=30,
        textColor=blue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=black
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=8,
        textColor=black
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        leftIndent=0.25*inch
    )
    
    def clean_text_for_pdf(text):
        import re
        text = re.sub(r'<b>([^<]*)<b>', r'<b>\1</b>', text)
        text = re.sub(r'<([^/>]+)>([^<]*)<(?!/\1)', r'<\1>\2</\1>', text)
        text = text.replace('<', '&lt;').replace('>', '&gt;')
        text = re.sub(r'&lt;b&gt;([^&]*)&lt;/b&gt;', r'<b>\1</b>', text)
        text = re.sub(r'&lt;i&gt;([^&]*)&lt;/i&gt;', r'<i>\1</i>', text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', text)
        text = re.sub(r'#+\s*', '', text)
        return text
    
    content = []
    
    content.append(Paragraph("Sustainability Analysis Report", title_style))
    content.append(Spacer(1, 20))
    
    content.append(Paragraph("Company Idea", heading_style))
    content.append(Paragraph(clean_text_for_pdf(company_idea), body_style))
    content.append(Spacer(1, 20))
    
    content.append(Paragraph("Sustainability Analysis", heading_style))
    
    cleaned_analysis = clean_text_for_pdf(analysis)
    analysis_paragraphs = cleaned_analysis.split('\n\n')
    
    for para in analysis_paragraphs:
        para = para.strip()
        if para:
            if para.startswith('###'):
                clean_heading = para.replace('###', '').strip()
                content.append(Paragraph(clean_text_for_pdf(clean_heading), subheading_style))
            elif para.startswith('##'):
                clean_heading = para.replace('##', '').strip()
                content.append(Paragraph(clean_text_for_pdf(clean_heading), heading_style))
            elif len(para) > 200:
                content.append(Paragraph(clean_text_for_pdf(para), body_style))
            else:
                content.append(Paragraph(clean_text_for_pdf(para), body_style))
            content.append(Spacer(1, 6))
    
    content.append(Spacer(1, 30))
    
    content.append(Paragraph("Detailed Sources and References", heading_style))
    
    total_docs = sum(len(docs) for docs in retrieved_docs.values())
    total_kbs = len([kb for kb, docs in retrieved_docs.items() if docs])
    
    content.append(Paragraph(
        f"This analysis was generated using {total_docs} source documents from {total_kbs} knowledge bases.",
        body_style
    ))
    content.append(Spacer(1, 15))
    
    for source_name, docs in retrieved_docs.items():
        if docs:
            kb_friendly_name = pipeline.knowledge_base_names.get(source_name, source_name)
            content.append(Paragraph(f"{kb_friendly_name} ({len(docs)} documents)", subheading_style))
            
            for i, doc in enumerate(docs[:3], 1):
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                source_id = metadata.get('source', metadata.get('id', f'Document {i}'))
                title = metadata.get('title', metadata.get('name', source_id))
                
                content.append(Paragraph(f"<b>{clean_text_for_pdf(title)}</b>", body_style))
                
                doc_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                preview = doc_content[:300] + "..." if len(doc_content) > 300 else doc_content
                content.append(Paragraph(clean_text_for_pdf(preview), body_style))
                content.append(Spacer(1, 10))
            
            content.append(Spacer(1, 15))
    
    content.append(PageBreak())
    content.append(Paragraph("Analysis Metadata", heading_style))
    
    metadata_items = [
        f"Generation Time: {analysis_time:.2f} seconds",
        f"Model Used: Llama-3.2-1B-Instruct",
        f"Total Documents Retrieved: {total_docs}",
        f"Knowledge Bases Used: {total_kbs}",
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    ]
    
    for item in metadata_items:
        content.append(Paragraph(f"• {item}", body_style))
    
    content.append(Spacer(1, 30))
    content.append(Paragraph(
        "This report was generated with full source attribution to ensure transparency and accountability in the analysis process.",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=9, textColor=grey, alignment=1)
    ))
    
    try:
        doc.build(content)
        buffer.seek(0)
        return buffer
    except Exception:
        buffer = BytesIO()
        simple_doc = SimpleDocTemplate(buffer, pagesize=A4)
        simple_content = [
            Paragraph("Sustainability Analysis Report", title_style),
            Spacer(1, 20),
            Paragraph("Company Idea", heading_style),
            Paragraph(company_idea, body_style),
            Spacer(1, 20),
            Paragraph("Analysis", heading_style),
            Paragraph(analysis.replace('<', '&lt;').replace('>', '&gt;'), body_style),
        ]
        simple_doc.build(simple_content)
        buffer.seek(0)
        return buffer


def main():
    st.set_page_config(
        page_title="Sustainability Analysis Tool",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    logger = get_logger()
    logger.info("Streamlit app started")
    
    st.title("Sustainability Analysis Tool")
    st.subheader("Analyze your company ideas against multiple sustainability frameworks")
    
    with st.spinner("Loading AI models and knowledge bases..."):
        pipeline, logger = load_pipeline()
    
    if pipeline is None:
        st.error("Failed to load the analysis pipeline. Please check the logs.")
        logger.error("Pipeline loading failed, stopping app")
        return
    
    display_knowledge_bases(pipeline, logger)
    
    st.markdown("---")
    
    st.subheader("Chat with the Sustainability Advisor")
    
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
    
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(f"**Company Idea:** {message['content']}")
                else:
                    st.markdown(message["content"])
    
    user_input = st.chat_input(
        "Describe your company idea and press Enter...",
        key="chat_input"
    )
    
    if user_input or ("user_input" in st.session_state and st.session_state.user_input):
        if "user_input" in st.session_state and st.session_state.user_input:
            company_idea = st.session_state.user_input
            st.session_state.user_input = ""
        else:
            company_idea = user_input
            
        if company_idea and company_idea.strip():
            logger.info(f"New user message: {company_idea[:100]}...")
            
            st.session_state.messages.append({"role": "user", "content": company_idea})
            
            with st.chat_message("user"):
                st.write(f"**Company Idea:** {company_idea}")
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your idea against sustainability frameworks..."):
                    try:
                        start_time = time.time()
                        
                        retrieved_docs, retrieval_stats = adaptive_document_retrieval(pipeline, company_idea, relevance_threshold=0.3)
                        
                        context = pipeline.build_context(retrieved_docs)
                        prompt = pipeline.build_prompt(company_idea, context)
                        
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
                        
                        generated_ids = output[0][inputs["input_ids"].shape[1]:]
                        analysis = pipeline.tokenizer.decode(
                            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                        )
                        
                        analysis = pipeline._clean_response(analysis)
                        
                        end_time = time.time()
                        analysis_time = end_time - start_time
                        
                        logger.info(f"Analysis completed in {analysis_time:.2f} seconds")
                        
                        st.markdown("## Sustainability Analysis")
                        st.markdown(analysis)

                        md_report = f"""# Sustainability Analysis Report

## Company Idea
{company_idea}

## Analysis
{analysis}

---
*Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""

                        st.download_button(
                            label="📥 Download this analysis as Markdown",
                            data=md_report,
                            file_name=f"sustainability_analysis_{int(time.time())}.md",
                            mime="text/markdown",
                            key=f"download_md_inline_{len(st.session_state.messages)}",
                            use_container_width=True,
                        )
                        
                        total_docs = sum(len(docs) for docs in retrieved_docs.values())
                        active_kbs = len([kb for kb, docs in retrieved_docs.items() if docs])
                        
                        st.session_state.last_analysis = {
                            'company_idea': company_idea,
                            'analysis': analysis,
                            'analysis_time': analysis_time,
                            'total_docs': total_docs,
                            'active_kbs': active_kbs,
                            'retrieved_docs': retrieved_docs,
                            'retrieval_stats': retrieval_stats
                        }
                        
                        display_sources_with_stats(retrieved_docs, retrieval_stats, pipeline, logger)
                        
                        st.markdown("---")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Analysis Time", f"{analysis_time:.1f}s")
                        with col2:
                            st.metric("Knowledge Bases", f"{active_kbs}")
                        with col3:
                            st.metric("Documents Used", f"{total_docs}")
                        with col4:
                            st.metric("Model", "Llama-3.2-1B")
                        
                        with st.expander("📋 Download Detailed Report with Sources"):
                            st.info("Get a comprehensive report including all source documents and metadata")
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                detailed_report = create_detailed_report(company_idea, analysis, retrieved_docs, pipeline, analysis_time)
                                st.download_button(
                                    label="Download Detailed Markdown",
                                    data=detailed_report,
                                    file_name=f"detailed_sustainability_analysis_{int(time.time())}.md",
                                    mime="text/markdown",
                                    key=f"download_detailed_md_{len(st.session_state.messages)}",
                                    use_container_width=True
                                )
                            
                            with col_b:
                                if REPORTLAB_AVAILABLE:
                                    try:
                                        pdf_buffer = create_pdf_report(company_idea, analysis, retrieved_docs, pipeline, analysis_time)
                                        st.download_button(
                                            label="Download PDF Report",
                                            data=pdf_buffer.getvalue(),
                                            file_name=f"sustainability_analysis_{int(time.time())}.pdf",
                                            mime="application/pdf",
                                            key=f"download_pdf_{len(st.session_state.messages)}",
                                            use_container_width=True
                                        )
                                    except Exception as e:
                                        st.error(f"PDF generation failed: {str(e)}")
                                        logger.error(f"PDF generation error: {str(e)}")
                                else:
                                    st.info("Install reportlab for PDF: pip install reportlab")
                        
                        sources_summary = ""
                        if retrieved_docs:
                            kb_counts = {pipeline.knowledge_base_names.get(kb, kb): len(docs) 
                                       for kb, docs in retrieved_docs.items() if docs}
                            sources_list = [f"{kb} ({count})" for kb, count in kb_counts.items()]
                            sources_summary = f"\n\n**Sources Used:** {', '.join(sources_list)}"
                        
                        response = f"## Sustainability Analysis\n\n{analysis}{sources_summary}\n\n---\n*Analysis completed in {analysis_time:.1f} seconds using {total_docs} source documents*"
                        
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        error_msg = f"Analysis failed: {str(e)}"
                        st.error(error_msg)
                        logger.error(f"Analysis failed: {str(e)}")
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

            st.rerun()
    
    st.markdown("---")
    # Add floating download button if analysis exists
    if hasattr(st.session_state, 'last_analysis') and st.session_state.last_analysis:
        analysis_data = st.session_state.last_analysis
        
        # Create basic report content
        basic_report = f"""# Sustainability Analysis Report
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Company Idea
{analysis_data['company_idea']}

## Analysis
{analysis_data['analysis']}

---
*Analysis completed in {analysis_data['analysis_time']:.1f} seconds using {analysis_data['total_docs']} source documents from {analysis_data['active_kbs']} knowledge bases*

**Model Used:** Llama-3.2-1B-Instruct
"""
        
        st.markdown("""
        <style>
        .floating-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 999;
        }
        </style>
        """, unsafe_allow_html=True)
        
        floating_container = st.container()
        with floating_container:
            st.markdown('<div class="floating-container">', unsafe_allow_html=True)
            st.download_button(
                label="📥 Download Report",
                data=basic_report,
                file_name=f"sustainability_analysis_{int(time.time())}.md",
                mime="text/markdown",
                key="floating_download_btn",
                help="Download the complete analysis as a Markdown file",
                type="primary"
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Built with Streamlit • Powered by Multi-Vector Store RAG • AI Sustainability Advisor</p>
        <p><small>Chat interface for analyzing company ideas against UN SDGs, EU Taxonomy, CSRD, and UN 2030 Agenda</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
