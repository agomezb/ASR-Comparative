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
    # 1. Load and Validate Ground Truth
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth_list = json.load(f)

    # Validate that no ground truth text is empty
    for item in ground_truth_list:
        if 'text' not in item or not str(item['text']).strip():
            raise ValueError(f"Invalid ground truth: 'text' field is empty or missing for id '{item.get('id', 'N/A')}'.")
    
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
    def calculate_wer_measures(row):
        reference = row['reference']
        hypothesis = str(row['text_normalized']) if pd.notna(row['text_normalized']) else ""
        
        if not reference:
            return pd.Series([None, 0, 0], index=['wer', 'errors', 'reference_words'])
        
        # Use jiwer.process_words for jiwer versions 3.0.0+
        output = jiwer.process_words(reference, hypothesis)
        
        errors = output.substitutions + output.deletions + output.insertions
        reference_words = output.hits + output.substitutions + output.deletions

        wer = output.wer
        
        return pd.Series([wer, errors, reference_words, output.substitutions, output.deletions, output.insertions], 
                         index=['wer', 'errors', 'reference_words', 'substitutions', 'deletions', 'insertions'])

    dataset_out[['wer', 'errors', 'reference_words', 'substitutions', 'deletions', 'insertions']] = dataset_out.apply(calculate_wer_measures, axis=1)

    # 4. Calculate Global WER
    # Filter for rows where we have a valid reference
    valid_data = dataset_out.dropna(subset=['wer'])
    
    if valid_data.empty:
        return 0.0, dataset_out

    # Calculate global WER based on total errors and words
    total_errors = valid_data['errors'].sum()
    total_reference_words = valid_data['reference_words'].sum()
    
    global_wer = total_errors / total_reference_words if total_reference_words > 0 else 0.0
    
    return global_wer, dataset_out