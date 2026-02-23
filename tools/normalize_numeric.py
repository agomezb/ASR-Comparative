#!/usr/bin/env python3
import re
from typing import Dict

# ==============================================================================
# 3. NORMALIZADOR NUMÉRICO (Experimental - Dígitos)
# ==============================================================================
class TextNormalizerNumeric:
    def __init__(self, text: str = ""):
        self.text = text
        self.technical_replacements = {
            "tb": "terabyte",
            "1tb": "un terabyte",
            "1 tb": "un terabyte",
            "1 terabyte": "un terabyte",
            "i 7": "i7", "a 4": "a4", "x g": "xg", "f a": "fa"
        }
        self.oov_replacements = {
            "compu facil": "compufacil", "compu fácil": "compufacil", "compufácil": "compufacil",
            "tecno sis": "tecnosys", "tecno sys": "tecnosys", "tecnosis": "tecnosys",
            "andina corp": "andinacorp",
            "dura disco": "duradisco",
            "dulcesideas": "dulces ideas", 
            "papelmundo": "papel mundo"
        }
        self.numerals = {
            "cero": "0", "uno": "1", "dos": "2", "tres": "3",
            "cuatro": "4", "cinco": "5", "seis": "6", "siete": "7", "ocho": "8",
            "nueve": "9", "diez": "10", "cincuenta": "50"
        }
    
    def set_text(self, text: str):
        self.text = text if text else ""
        return self

    def to_lowercase(self):
        self.text = self.text.lower()
        return self

    def remove_punctuation(self):
        self.text = re.sub(r'[^\w\s]', ' ', self.text).replace('_', ' ')
        return self

    def convert_words_to_numbers(self):
        for word, digit in self.numerals.items():
            self.text = re.sub(r'\b' + word + r'\b', digit, self.text)
        return self

    def separate_codes_from_letters(self):
        def is_code(s): return (s.startswith('0') and len(s) > 1) or len(s) >= 4
        def repl(m): return f"{m.group(1)} {m.group(2)}" if is_code(m.group(2)) else m.group(0)
        
        self.text = re.sub(r'([a-zA-Záéíóúñ])(\d+)', repl, self.text)
        self.text = re.sub(r'(\d+)([a-zA-Záéíóúñ])', lambda m: f"{m.group(1)} {m.group(2)}" if is_code(m.group(1)) else m.group(0), self.text)
        return self

    def process_numeric_sequences(self):
        # Unify fragments: 80 20 25 -> 802025, 80-20-25 -> 802025
        self.text = re.sub(r'(?<=\d)[\s-]+(?=\d)', '', self.text)
        return self

    def apply_replacements(self):
        for k, v in {**self.technical_replacements, **self.oov_replacements}.items():
            self.text = re.sub(r'\b' + re.escape(k) + r'\b', v, self.text, flags=re.IGNORECASE)
        return self

    def clean_spaces(self):
        self.text = re.sub(r'\s+', ' ', self.text).strip()
        return self

    def build(self) -> str:
        return self.text

    def normalize(self, text: str) -> str:
        return (self.set_text(text)
                .to_lowercase()
                .remove_punctuation()
                .convert_words_to_numbers()
                .separate_codes_from_letters()
                .process_numeric_sequences()
                .apply_replacements()
                .clean_spaces()
                .build())
