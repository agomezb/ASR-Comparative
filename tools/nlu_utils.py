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
        scenario_id = int(row['audio'])
    
         # Fetch the corresponding rule
        if not self.rules or scenario_id not in self.rules:
            raise ValueError(f"Scenario ID {scenario_id} not found in rules.")

        rule = self.rules[scenario_id]
        text = row['text_normalized']

        # 1. Intent Logic
        intent_expected = rule["intent"]
        
        # Check if keywords exists in text
        intent_success = self._validate_intent(text, rule)
        intent_predicted = intent_expected if intent_success else "no_match"

        # 2. Slot Logic
        slot_evaluation = self._evaluate_slots(text, rule)
        slots_success = (slot_evaluation["nlu_hits"] == slot_evaluation["nlu_total"])

        # 3. Overall NLU Success
        nlu_success = intent_success and slots_success


        return pd.Series({
            # intent details
            "intent_expected": intent_expected,
            "intent_predicted": intent_predicted,
            "intent_success": intent_success,
            #nlu slots details
            "nlu_hits": slot_evaluation["nlu_hits"],
            "nlu_misses": slot_evaluation["nlu_misses"],
            "nlu_recall": slot_evaluation["nlu_recall"],
            "nlu_total": slot_evaluation["nlu_total"],
            "nlu_details": slot_evaluation["nlu_details"],
            # final results
            "slots_success": slots_success,
            "nlu_success": nlu_success
        })

    def _validate_intent(self, text: str, rule: Dict) -> bool:
        """
        Validate the intent based on keywords in the text.
        """
        keywords = rule["keywords"]
        matches = 0
        for keyword in keywords:
            score = fuzz.partial_ratio(keyword, text)
            if score > 80:
                matches += 1
        return matches == len(keywords)

    def _evaluate_slots(self, text_norm: str, rule: Dict) -> Dict[str, str]:
        expected_slots_list = rule["slots"]
        hits = 0
        misses = 0
        details = []
        
        for slot in expected_slots_list:
            # Extraemos clave y valor del nuevo formato de lista
            key_name = slot["key"]
            target_val = str(slot["val"]).lower()
            
            # --- Lógica de Búsqueda Difusa ---
            
            # Caso A: Palabras cortas (ej: "dos", "5", "A4")
            # partial_ratio es peligroso aquí (encuentra "dos" dentro de "todos").
            # Usamos token_set_ratio que busca coincidencias de tokens completos.
            if len(target_val) <= 6:
                score = fuzz.token_set_ratio(target_val, text_norm)
                
            # Caso B: Frases o palabras largas (ej: "monitores led", "compufacil")
            # partial_ratio es ideal porque tolera que el texto esté cortado o rodeado de ruido.
            else:
                score = fuzz.partial_ratio(target_val, text_norm)
                
            # --- Evaluación ---
            
            # Umbral 80: Tolera pequeños errores tipográficos (ej: "compufasil")
            if score >= 80:
                hits += 1
                details.append(f"{key_name}: '{target_val}' encontrado (Score: {score})")
            else:
                misses += 1
                details.append(f"{key_name}: '{target_val}' perdido (Score: {score})")

        # Cálculo final de la métrica
        total_expected = hits + misses
        survival_rate = hits / total_expected if total_expected > 0 else 0.0
        
        return {
            "nlu_hits": hits,
            "nlu_misses": misses,
            "nlu_total": total_expected,
            "nlu_recall": round(survival_rate, 4), # métrica principal
            "nlu_details": str(details)
        }
