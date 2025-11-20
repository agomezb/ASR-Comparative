import json
import pandas as pd
import jiwer
import re

def calculate_wer_from_dataframe(dataset: pd.DataFrame, ground_truth_path: str):
    """
    Calculates the Global WER (Micro-Average) for a given dataset against a ground truth JSON.

    Args:
        dataset (pd.DataFrame): The dataset containing ASR transcriptions. 
                                Must contain 'audio' and 'text_normalized' columns.
        ground_truth_path (str): Path to the JSON file containing ground truth texts.

    Returns:
        tuple: (Global WER (float), Modified DataFrame (pd.DataFrame))
    """
    # 1. Load Ground Truth
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth_list = json.load(f)
    
    # Convert to dictionary for fast lookup: {id: text}
    ground_truth_map = {item['id']: item['text'] for item in ground_truth_list}

    # 2. Create 'reference' column
    # We use a copy to avoid SettingWithCopyWarning if dataset is a slice
    dataset_out = dataset.copy()

    def get_reference(row):
        audio_val = str(row['audio'])
        return ground_truth_map.get(audio_val, "")

    dataset_out['reference'] = dataset_out.apply(get_reference, axis=1)

    # 3. Create 'wer' column (Row-level WER)
    def calculate_row_wer(row):
        reference = row['reference']
        hypothesis = str(row['text_normalized']) if pd.notna(row['text_normalized']) else ""
        
        if not reference:
            return None
        
        return jiwer.wer(reference, hypothesis)

    dataset_out['wer'] = dataset_out.apply(calculate_row_wer, axis=1)

    # 4. Calculate Global WER
    # Filter for rows where we have a valid reference
    valid_data = dataset_out[dataset_out['reference'] != ""]
    
    if valid_data.empty:
        return 0.0, dataset_out

    # Get lists from columns
    reference_list = valid_data['reference'].tolist()
    hypothesis_list = valid_data['text_normalized'].fillna("").astype(str).tolist()

    global_wer = jiwer.wer(reference_list, hypothesis_list)
    
    return global_wer, dataset_out
