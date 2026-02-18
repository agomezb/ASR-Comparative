#!/usr/bin/env python3
import re
from typing import Dict

# ==============================================================================
# 3. NORMALIZADOR NUMÉRICO (Experimental - Dígitos)
# ==============================================================================
class TextNormalizerNumeric:
    def __init__(self):
        self.technical_replacements = {
            "tb": "terabyte", "1tb": "1 terabyte", "i 7": "i7", "a 4": "a4", "x g": "xg", "f a": "fa"}
        self.oov_replacements = {
            "compu facil": "compufacil", "compu fácil": "compufacil", "compufácil": "compufacil",
            "tecno sis": "tecnosys", "techno sys": "tecnosys", "tecno sys": "tecnosys", "tecnosis": "tecnosys",
            "andina corp": "andinacorp", "andina corp.": "andinacorp", "dura disco": "duradisco",
            "dulcesideas": "dulces ideas", "papelmundo": "papel mundo"
        }
        self.numerals = {
            "cero": "0", "un": "1", "uno": "1", "una": "1", "dos": "2", "tres": "3",
            "cuatro": "4", "cinco": "5", "seis": "6", "siete": "7", "ocho": "8",
            "nueve": "9", "diez": "10", "cincuenta": "50"
        }
    
    def normalize(self, text: str) -> str:
        if not text: return ""
        
        # 1. minúsculas
        text = text.lower()
        
        # 2. limpieza inicial de puntuación
        text = self._remove_punctuation(text)
        
        # 3. conversión palabras a números
        # transforma texto a dígito: cinco -> 5
        for word, digit in self.numerals.items():
            text = re.sub(r'\b' + word + r'\b', digit, text)
            
        # 4. separación de códigos y letras
        # separa letras de números solo si son códigos largos (fa4095 -> fa 4095), respeta i7
        text = self._separate_codes_from_letters(text)
        
        # 5. procesamiento de secuencias numéricas
        # unifica fragmentos (85 20 25 -> 852025) para validar si es código y luego formatea (8 5 2 0 2 5)
        text = self._process_numeric_sequences(text)
        
        # 6. reemplazos de texto
        # aplica correcciones de marca y re-pega formatos técnicos: i 7 -> i7
        for k, v in {**self.technical_replacements, **self.oov_replacements}.items():
            text = re.sub(r'\b' + re.escape(k) + r'\b', v, text, flags=re.IGNORECASE)
            
        # 7. limpieza final de espacios
        return re.sub(r'\s+', ' ', text).strip()

    def _remove_punctuation(self, text):
        return re.sub(r'[^\w\s]', ' ', text).replace('_', ' ')

    def _separate_codes_from_letters(self, text):
        def is_code(s): return (s.startswith('0') and len(s) > 1) or len(s) >= 4
        def repl(m): return f"{m.group(1)} {m.group(2)}" if is_code(m.group(2)) else m.group(0)
        text = re.sub(r'([a-zA-Záéíóúñ])(\d+)', repl, text)
        return re.sub(r'(\d+)([a-zA-Záéíóúñ])', lambda m: f"{m.group(1)} {m.group(2)}" if is_code(m.group(1)) else m.group(0), text)

    def _process_numeric_sequences(self, text):
        # Unify fragments: 80 20 25 -> 802025, 80-20-25 -> 802025
        return re.sub(r'(?<=\d)[\s-]+(?=\d)', '', text)
