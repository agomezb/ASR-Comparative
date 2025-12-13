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

    def _generate_default_result(self):
        """
        Generate a default result dictionary for unknown or error cases.
        """
        return pd.Series({
            "intent_expected": "unknown",
            "intent_predicted": "error",
            "intent_success": False,
            "ner_tp": 0,
            "ner_fn": 0,
            "ner_fp": 0,
            "slots_success": False,
            "nlu_success": False
        })


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
            return self._generate_default_result()

        if not self.rules or scenario_id not in self.rules:
            return self._generate_default_result()

        rule = self.rules[scenario_id]
        text = str(row['text_normalized']).lower() if pd.notna(row['text_normalized']) else ""

        # 1. Intent Logic
        intent_expected = rule["intent"]
        
        # Check if keywords exists in text
        intent_predicted = self._predict_intent(text, self.rules)
        intent_success = (intent_predicted == intent_expected)

        # 2. Slot Logic
        slot_evaluation = self._evaluate_slots(text, rule)
        slots_success = (slot_evaluation["ner_tp"] == slot_evaluation["ner_total"])

        # 3. Overall NLU Success
        nlu_success = intent_success and slots_success

        return pd.Series({
            "intent_expected": intent_expected,
            "intent_predicted": intent_predicted,
            "intent_success": intent_success,
            "ner_tp": slot_evaluation["ner_tp"],
            "ner_fn": slot_evaluation["ner_fn"],
            "ner_fp": slot_evaluation["ner_fp"],
            "slots_success": slots_success,
            "nlu_success": nlu_success
        })

    def _predict_intent(self, text: str, rules: Dict[int, Dict]) -> str:
        """
        Predict the intent based on keywords in the text.
        """
        for rule_data in rules.values():
            matches = 0
            for keyword in rule_data["keywords"]:
                score = fuzz.partial_ratio(keyword, text)
                if score > 80:
                    matches += 1
            if matches == len(rule_data["keywords"]):
                return rule_data["intent"] 
        return "no_match"

    def _evaluate_slots(self, text: str, rule: Dict) -> Dict[str, str]:
        """
        Predict slots based on fuzzy matching.
        """
        slot_expected = rule["slots"]
        tp = 0
        fp = 0
        fn = 0
        slots_detail = []
        for slot_key, slot_value in slot_expected.items():
            score = fuzz.partial_ratio(slot_value, text)
            if score > 80:
                tp += 1
                slots_detail.append(f"{slot_key}:OK({slot_value})")
            else:
                fn += 1
                slots_detail.append(f"{slot_key}:MISS (Esp:{slot_value})")
        total_expected = len(slot_expected)
        assert tp + fn == total_expected, "Slot counting error"
        return {
                    "ner_tp": tp, 
                    "ner_fp": fp, 
                    "ner_fn": fn,
                    "ner_total": total_expected,
                    "ner_details": str(slots_detail)
                }

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
