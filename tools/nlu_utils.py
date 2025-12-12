import pandas as pd
from fuzzywuzzy import fuzz
import json
from typing import Dict

class NLUEvaluator:
    """
    A class to evaluate ASR transcriptions based on business logic rules for intents and slots.
    """

    def __init__(self, rules_path: str = "tools/nlu_rules.json"):
        """
        Initializes the evaluator by loading rules from a JSON file.

        Args:
            rules_path (str): The path to the JSON file containing the NLU rules.
        """
        self.rules = self._load_rules(rules_path)

    def _load_rules(self, file_path: str) -> Dict[int, Dict]:
        """
        Loads NLU rules from a JSON file.
        Keys are converted from string to integer.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                rules_as_str_keys = json.load(f)
                return {int(k): v for k, v in rules_as_str_keys.items()}
        except FileNotFoundError:
            print(f"Error: Rules file not found at {file_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}")
            return {}

    def evaluate_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the dataset and add NLU evaluation columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing 'audio' (as ID) and 'text_normalized'.
            
        Returns:
            pd.DataFrame: DataFrame with additional evaluation columns.
        """
        # Create a copy to avoid SettingWithCopy warnings
        result_df = df.copy()
        
        # Apply evaluation row by row
        evaluation_results = result_df.apply(self._evaluate_row, axis=1)
        
        # Concatenate results
        result_df = pd.concat([result_df, evaluation_results], axis=1)
        
        return result_df

    def _evaluate_row(self, row):
        """
        Evaluate a single row against the rules.
        """
        # Extract ID from 'audio' column. Assuming 'audio' column contains the ID.
        # The prompt says: "Each row has an `id` (integer from 1 to 15)". 
        # In the provided dataframe sample, 'audio' column seems to hold the ID (1, 2, 3...).
        try:
            scenario_id = int(row['audio'])
        except (ValueError, KeyError):
            # Handle cases where ID might not be parseable or missing
            return pd.Series({
                "intent_expected": "unknown",
                "intent_predicted": "error",
                "intent_success": False,
                "slots_found_count": 0,
                "slots_total_count": 0,
                "slots_success": False,
                "nlu_success": False
            })

        if not self.rules or scenario_id not in self.rules:
             return pd.Series({
                "intent_expected": "unknown",
                "intent_predicted": "error",
                "intent_success": False,
                "slots_found_count": 0,
                "slots_total_count": 0,
                "slots_success": False,
                "nlu_success": False
            })

        rule = self.rules[scenario_id]
        text = str(row['text_normalized']).lower() if pd.notna(row['text_normalized']) else ""

        # 1. Intent Logic
        intent_expected = rule["intent"]
        keywords = rule["keywords"]
        
        # Check if *any* keyword exists in text
        intent_match = any(keyword in text for keyword in keywords)
        
        intent_predicted = intent_expected if intent_match else "no_match"
        intent_success = (intent_predicted == intent_expected)

        # 2. Slot Logic
        required_slots = rule["slots"]
        slots_found = 0
        
        for slot_key, slot_value in required_slots.items():
            # Fuzzy match
            # Use fuzz.partial_ratio > 85
            score = fuzz.partial_ratio(slot_value, text)
            if score > 85:
                slots_found += 1
        
        slots_total = len(required_slots)
        slots_success = (slots_found == slots_total)

        # 3. Overall NLU Success
        nlu_success = intent_success and slots_success

        return pd.Series({
            "intent_expected": intent_expected,
            "intent_predicted": intent_predicted,
            "intent_success": intent_success,
            "slots_found_count": slots_found,
            "slots_total_count": slots_total,
            "slots_success": slots_success,
            "nlu_success": nlu_success
        })

if __name__ == "__main__":
    # Usage example with dummy DataFrame
    data = {
        'audio': [1, 2, 3, 1],
        'text_normalized': [
            "genera una cotización para el cliente compufacil cantidad cinco", # Perfect match ID 1
            "presupuesto para tecnosys cantidad diez", # Perfect match ID 2
            "oferta comercial para carla modelo equis ge", # Perfect match ID 3
            "cotizacion para cliente compufacil" # Missing 'cinco' for ID 1
        ]
    }
    df_dummy = pd.DataFrame(data)
    
    # The evaluator now loads rules from 'tools/nlu_rules.json' by default
    evaluator = NLUEvaluator()
    df_result = evaluator.evaluate_dataset(df_dummy)
    
    print("Usage Example Results:")
    print(df_result[['audio', 'text_normalized', 'intent_success', 'slots_success', 'nlu_success']])
