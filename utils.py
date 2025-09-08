# utils.py
import csv
from collections import defaultdict

def load_structured_data(file_path: str):
    """
    Load data from a binary CSV, merging sentences for each disease.
    If a document is too long, it's split into smaller chunks only at newlines.
    """
    disease_sentences_map = defaultdict(list)

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        symptom_names = header[1:]
        
        for row in reader:
            disease = row[0].strip()
            symptom_indicators = row[1:]
            
            present_symptoms = []
            for i, indicator in enumerate(symptom_indicators):
                if indicator == '1':
                    present_symptoms.append(symptom_names[i])
            
            if present_symptoms:
                symptoms_text = ", ".join(present_symptoms)
                sentence = f"{disease} can present with symptoms such as {symptoms_text}."
                disease_sentences_map[disease].append(sentence)
            
    documents = []
    ids = []
    max_chars_per_chunk = 6000

    for disease, sentences_list in disease_sentences_map.items():
        full_text = "\n".join(sentences_list)
        
        if len(full_text) <= max_chars_per_chunk:
            documents.append(full_text)
            ids.append(disease)
        else:
            print(f"Document for '{disease}' is too long, splitting at newlines...")
            remaining_text = full_text
            chunk_num = 1
            while remaining_text:
                split_point = min(len(remaining_text), max_chars_per_chunk)
                
                # This line is now changed to only look for newlines.
                safe_split_point = remaining_text.rfind('\n', 0, split_point)
                
                if safe_split_point == -1: # If no newline is found, make a hard cut
                    safe_split_point = split_point

                chunk_text = remaining_text[:safe_split_point]
                documents.append(chunk_text)
                ids.append(f"{disease}-{chunk_num}")
                
                remaining_text = remaining_text[safe_split_point:].lstrip()
                chunk_num += 1

    return documents, ids