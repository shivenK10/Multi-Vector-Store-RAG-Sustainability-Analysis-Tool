import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os

class FAISSVectorStoreBuilder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def extract_text_chunks(self, data, file_name):
        chunks = []
        
        if file_name == "un_2030_agenda.json":
            chunks.extend(self._extract_un_agenda_chunks(data))
        elif file_name == "sdgs.json":
            chunks.extend(self._extract_sdgs_chunks(data))
        elif file_name == "eu_taxonomy_user_guide.json":
            chunks.extend(self._extract_eu_taxonomy_chunks(data))
        elif file_name == "csrd_fact_sheet.json":
            chunks.extend(self._extract_csrd_chunks(data))
        
        return chunks
    
    def _extract_un_agenda_chunks(self, data):
        chunks = []
        

        chunks.append({
            "text": f"{data.get('title', '')}: {data.get('overall_summary', '')}",
            "metadata": {
                "source": "un_2030_agenda",
                "type": "overview",
                "framework_id": data.get('framework_id', ''),
                "chunk_id": "overview"
            }
        })
        

        for i, principle in enumerate(data.get('core_principles', [])):
            chunks.append({
                "text": f"Core Principle: {principle}",
                "metadata": {
                    "source": "un_2030_agenda",
                    "type": "principle",
                    "principle_id": f"principle_{i+1}",
                    "chunk_id": f"principle_{i+1}"
                }
            })
        

        for pillar in data.get('pillars', []):
            text = f"Pillar {pillar.get('title', '')}: {pillar.get('summary', '')} Focus areas: {', '.join(pillar.get('focus_areas', []))}"
            chunks.append({
                "text": text,
                "metadata": {
                    "source": "un_2030_agenda",
                    "type": "pillar",
                    "pillar_id": pillar.get('pillar_id', ''),
                    "related_sdgs": pillar.get('related_sdgs', []),
                    "chunk_id": f"pillar_{pillar.get('pillar_id', '')}"
                }
            })
        

        for sdg in data.get('sdg_overview', []):
            text = f"SDG {sdg.get('sdg_id', '')}: {sdg.get('sdg_name', '')} - {sdg.get('headline', '')} Main themes: {', '.join(sdg.get('main_themes', []))}"
            chunks.append({
                "text": text,
                "metadata": {
                    "source": "un_2030_agenda",
                    "type": "sdg_overview",
                    "sdg_id": sdg.get('sdg_id', ''),
                    "sdg_name": sdg.get('sdg_name', ''),
                    "chunk_id": f"sdg_overview_{sdg.get('sdg_id', '')}"
                }
            })
        
        return chunks
    
    def _extract_sdgs_chunks(self, data):
        chunks = []
        
        for sdg in data:

            text = f"SDG {sdg.get('sdg_id', '')}: {sdg.get('sdg_name', '')} - {sdg.get('headline', '')} Summary: {sdg.get('summary', '')}"
            chunks.append({
                "text": text,
                "metadata": {
                    "source": "sdgs",
                    "type": "sdg_main",
                    "sdg_id": sdg.get('sdg_id', ''),
                    "sdg_name": sdg.get('sdg_name', ''),
                    "category": sdg.get('category', []),
                    "chunk_id": f"sdg_{sdg.get('sdg_id', '')}_main"
                }
            })
            

            for target in sdg.get('targets', []):
                target_text = f"SDG {sdg.get('sdg_id', '')} Target: {target.get('target_description', '')} Indicators: {', '.join(target.get('key_indicators', []))}"
                chunks.append({
                    "text": target_text,
                    "metadata": {
                        "source": "sdgs",
                        "type": "target",
                        "sdg_id": sdg.get('sdg_id', ''),
                        "target_id": target.get('target_id', ''),
                        "indicators": target.get('key_indicators', []),
                        "chunk_id": f"sdg_{sdg.get('sdg_id', '')}_target_{target.get('target_id', '')}"
                    }
                })
            

            for i, example in enumerate(sdg.get('business_examples', [])):
                example_text = f"SDG {sdg.get('sdg_id', '')} Business Example: {example.get('description', '')} Sectors: {', '.join(example.get('possible_sectors', []))}"
                chunks.append({
                    "text": example_text,
                    "metadata": {
                        "source": "sdgs",
                        "type": "business_example",
                        "sdg_id": sdg.get('sdg_id', ''),
                        "example_id": i,
                        "sectors": example.get('possible_sectors', []),
                        "chunk_id": f"sdg_{sdg.get('sdg_id', '')}_example_{i}"
                    }
                })
        
        return chunks
    
    def _extract_eu_taxonomy_chunks(self, data):
        chunks = []
        chunks.append({
            "text": f"{data.get('title', '')}: {data.get('overall_summary', '')}",
            "metadata": {
                "source": "eu_taxonomy",
                "type": "overview",
                "framework_id": data.get('framework_id', ''),
                "chunk_id": "overview"
            }
        })
        

        for obj in data.get('environmental_objectives', []):
            text = f"Environmental Objective: {obj.get('name', '')} - {obj.get('description', '')}"
            chunks.append({
                "text": text,
                "metadata": {
                    "source": "eu_taxonomy",
                    "type": "environmental_objective",
                    "objective_id": obj.get('objective_id', ''),
                    "chunk_id": f"objective_{obj.get('objective_id', '')}"
                }
            })
        

        for condition in data.get('alignment_conditions', []):
            text = f"Alignment Condition - {condition.get('title', '')}: {condition.get('summary', '')}"
            chunks.append({
                "text": text,
                "metadata": {
                    "source": "eu_taxonomy",
                    "type": "alignment_condition",
                    "condition_id": condition.get('condition_id', ''),
                    "chunk_id": f"condition_{condition.get('condition_id', '')}"
                }
            })
        

        for step in data.get('step_by_step_alignment_process', []):
            text = f"Step {step.get('step_number', '')}: {step.get('title', '')} - {step.get('description', '')}"
            chunks.append({
                "text": text,
                "metadata": {
                    "source": "eu_taxonomy",
                    "type": "alignment_step",
                    "step_number": step.get('step_number', ''),
                    "chunk_id": f"step_{step.get('step_number', '')}"
                }
            })
        
        return chunks
    
    def _extract_csrd_chunks(self, data):
        chunks = []
        chunks.append({
            "text": f"{data.get('title', '')}: {data.get('overall_summary', '')}",
            "metadata": {
                "source": "csrd",
                "type": "overview",
                "framework_id": data.get('framework_id', ''),
                "chunk_id": "overview"
            }
        })
        

        for phase in data.get('official_timeline', []):
            text = f"CSRD Timeline {phase.get('phase', '')}: {phase.get('companies', '')} - First reporting: {phase.get('first_reporting_year', '')}, Status: {phase.get('status', '')}"
            chunks.append({
                "text": text,
                "metadata": {
                    "source": "csrd",
                    "type": "timeline",
                    "phase": phase.get('phase', ''),
                    "reporting_year": phase.get('first_reporting_year', ''),
                    "chunk_id": f"timeline_{phase.get('phase', '')}"
                }
            })
        

        esrs_arch = data.get('esrs_architecture', {})
        for category, standards in esrs_arch.items():
            if isinstance(standards, list):
                for standard in standards:
                    if isinstance(standard, dict) and 'code' in standard:
                        text = f"ESRS {standard.get('code', '')}: {standard.get('title', '')} - Focus: {', '.join(standard.get('focus', []))}"
                        chunks.append({
                            "text": text,
                            "metadata": {
                                "source": "csrd",
                                "type": "esrs_standard",
                                "category": category,
                                "code": standard.get('code', ''),
                                "chunk_id": f"esrs_{standard.get('code', '')}"
                            }
                        })
        

        for i, feature in enumerate(data.get('key_features', [])):
            chunks.append({
                "text": f"CSRD Key Feature: {feature}",
                "metadata": {
                    "source": "csrd",
                    "type": "key_feature",
                    "feature_id": i,
                    "chunk_id": f"feature_{i}"
                }
            })
        
        return chunks
    
    def build_vector_store(self, json_file_path, output_dir="vector_stores"):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        file_name = os.path.basename(json_file_path)
        
        chunks = self.extract_text_chunks(data, file_name)
        
        if not chunks:
            return None
        
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        index = faiss.IndexFlatIP(self.dimension)
        
        faiss.normalize_L2(embeddings)
        
        index.add(embeddings.astype(np.float32))
        
        os.makedirs(output_dir, exist_ok=True)
        base_name = file_name.replace('.json', '')
        index_path = os.path.join(output_dir, f"{base_name}_faiss.index")
        metadata_path = os.path.join(output_dir, f"{base_name}_metadata.pkl")
        
        faiss.write_index(index, index_path)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(chunks, f)
        
        return index_path
    
    def build_all_stores(self, json_files, output_dir="vector_stores"):
        results = {}
        
        for json_file in json_files:
            try:
                store_path = self.build_vector_store(json_file, output_dir)
                if store_path:
                    results[os.path.basename(json_file)] = store_path
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        return results

def main():
    builder = FAISSVectorStoreBuilder()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_files = [
        os.path.join(current_dir, "un_2030_agenda.json"),
        os.path.join(current_dir, "sdgs.json"),
        os.path.join(current_dir, "eu_taxonomy_user_guide.json"),
        os.path.join(current_dir, "csrd_fact_sheet.json")
    ]
    existing_files = []
    for file_path in json_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            print(f"File not found: {file_path}")
    
    if not existing_files:
        print("No JSON files found!")
        return
    results = builder.build_all_stores(existing_files)
    print("\n" + "="*50)
    print("FAISS Vector Store Build Summary")
    print("="*50)
    
    for json_file, store_path in results.items():
        print(f"✓ {json_file} -> {store_path}")
    
    print(f"\nTotal vector stores created: {len(results)}")
    print("Vector stores saved in 'vector_stores/' directory")

if __name__ == "__main__":
    main()