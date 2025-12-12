import pandas as pd
from fuzzywuzzy import fuzz

class NLUEvaluator:
    """
    A class to evaluate ASR transcriptions based on business logic rules for intents and slots.
    """

    # Golden Rules Definition
    RULES = {
        1: {
            "intent": "crear_cotizacion",
            "keywords": ["cotización", "cotizacion"],
            "slots": {"cliente": "compufacil", "cantidad": "cinco"}
        },
        2: {
            "intent": "crear_presupuesto",
            "keywords": ["presupuesto"],
            "slots": {"cliente": "tecnosys", "cantidad": "diez"}
        },
        3: {
            "intent": "crear_oferta",
            "keywords": ["oferta", "comercial"],
            "slots": {"contacto": "carla", "modelo": "equis ge"}
        },
        4: {
            "intent": "crear_proforma",
            "keywords": ["proforma"],
            "slots": {"cliente": "andina", "modelo": "core i siete", "cantidad": "dos"}
        },
        5: {
            "intent": "buscar_factura",
            "keywords": ["busca", "factura"],
            "slots": {"cliente": "velasco"}
        },
        6: {
            "intent": "generar_factura",
            "keywords": ["genera", "factura"],
            "slots": {"serie": "ocho cinco dos cero dos cinco"}
        },
        7: {
            "intent": "verificar_pago",
            "keywords": ["verifica", "pagada"],
            "slots": {"factura": "efe a cuatro cero nueve cinco"}
        },
        8: {
            "intent": "consultar_inventario",
            "keywords": ["cuantas", "sillas"],
            "slots": {"ubicacion": "guayaquil"}
        },
        9: {
            "intent": "registrar_ingreso",
            "keywords": ["ingreso", "registra"],
            "slots": {"cantidad": "cincuenta", "item": "resmas"}
        },
        10: {
            "intent": "consultar_stock",
            "keywords": ["stock", "dame"],
            "slots": {"capacidad": "terabyte", "marca": "dura disco"}
        },
        11: {
            "intent": "ver_historial",
            "keywords": ["historial", "compras"],
            "slots": {"cliente": "terroso", "tiempo": "seis"}
        },
        12: {
            "intent": "crear_cliente",
            "keywords": ["crea", "cliente"],
            "slots": {"ruc": "cero nueve dos dos"}
        },
        13: {
            "intent": "info_contacto",
            "keywords": ["numero", "telefono","número","teléfono"],
            "slots": {"empresa": "dulces ideas"}
        },
        14: {
            "intent": "reporte_ventas",
            "keywords": ["reporte", "ventas"],
            "slots": {"vendedor": "daniel", "mes": "agosto"}
        },
        15: {
            "intent": "reporte_top_producto",
            "keywords": ["producto", "vendido"],
            "slots": {"tiempo": "trimestre"}
        }
    }

    def __init__(self):
        pass

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

        if scenario_id not in self.RULES:
             return pd.Series({
                "intent_expected": "unknown",
                "intent_predicted": "error",
                "intent_success": False,
                "slots_found_count": 0,
                "slots_total_count": 0,
                "slots_success": False,
                "nlu_success": False
            })

        rule = self.RULES[scenario_id]
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
    
    evaluator = NLUEvaluator()
    df_result = evaluator.evaluate_dataset(df_dummy)
    
    print("Usage Example Results:")
    print(df_result[['audio', 'text_normalized', 'intent_success', 'slots_success', 'nlu_success']])
