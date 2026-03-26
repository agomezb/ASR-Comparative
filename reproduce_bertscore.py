
from bert_score import score
import logging

try:
    cands = ["Hola mundo", "Esto es una prueba"]
    refs = ["Hola mundo", "Esta es una prueba"]

    print("Iniciando cálculo de BERTScore con PlanTL-GOB-ES/roberta-base-bne...")
    P, R, F1 = score(cands, refs, verbose=True, model_type="PlanTL-GOB-ES/roberta-base-bne")
    print(f"F1 scores: {F1}")
    print("Éxito!")

except Exception as e:
    print(f"Error detectado: {e}")
