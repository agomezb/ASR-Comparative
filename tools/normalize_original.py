#!/usr/bin/env python3
import re
from typing import Dict
from num2words import num2words

# ==============================================================================
# 1. NORMALIZADOR ORIGINAL (Full Rules - Fonético)
# ==============================================================================
class TextNormalizerOriginal:
    def __init__(self):
        self.custom_replacements = {
            "1 tb": "un terabyte", "tb": "terabyte", "gb": "gigabyte",
            "i 7": "i siete", "i 5": "i cinco", "i 3": "i tres",
            "a 4": "a cuatro", "a4": "a cuatro", "xg": "equis ge",
            "f a": "efe a", "fa": "efe a", "ruck": "ruc", "rug": "ruc",
            "compu facil": "compufacil", "compu fácil": "compufacil", "compufácil": "compufacil",
            "tecno sis": "tecnosys", "techno sys": "tecnosys", "tecno sys": "tecnosys", "tecnosis": "tecnosys",
            "andina corp": "andinacorp", "andina corp.": "andinacorp", "dura disco": "duradisco",
            "dulcesideas": "dulces ideas", "papelmundo": "papel mundo"
        }
    
    def normalize(self, text: str) -> str:
        if not text: return ""
        
        # 1. minúsculas
        text = text.lower()
        
        # 2. limpieza de puntuación segura
        # reemplazamos puntos y comas por espacios para proteger siglas como f.a.
        text = text.replace('.', ' ').replace(',', ' ')
        
        # 3. separación de letras y números
        # vital para modelos que pegan códigos: i7 -> i 7
        text = self._separate_letters_and_numbers(text)
        
        # 4. reemplazos de dominio
        # aplica correcciones fonéticas y de marcas: tecnosis -> tecnosys
        for k, v in self.custom_replacements.items():
            text = re.sub(r'\b' + re.escape(k) + r'\b', v, text, flags=re.IGNORECASE)
            
        # 5. unificación de dígitos
        # convierte 8 5 2 0 en 8520 para detectar códigos largos
        text = re.sub(r'(?<=\d)[\s-]+(?=\d)', '', text)
        
        # 6. conversión numérica contextual
        # decide si es código (dígito a dígito) o cantidad (cardinal)
        text = self._numbers_to_words(text)
        
        # 7. limpieza final destructiva
        # elimina todo lo que no sea letra o espacio
        text = re.sub(r'[^\w\s]', ' ', text).replace('_', ' ')
        
        # 8. ajustes gramaticales
        # corrige concordancia generada por num2words: uno monitor -> un monitor
        text = self._spanish_post_processing(text)
        
        # 9. limpieza de espacios
        return re.sub(r'\s+', ' ', text).strip()

    def _separate_letters_and_numbers(self, text):
        text = re.sub(r'([a-zA-Záéíóúñ])(\d)', r'\1 \2', text)
        return re.sub(r'(\d)([a-zA-Záéíóúñ])', r'\1 \2', text)

    def _numbers_to_words(self, text):
        def convert(m):
            s = m.group(0)
            # regla: si empieza con 0 o tiene 4+ dígitos -> código (dígito a dígito)
            if (s.startswith('0') and len(s) > 1) or len(s) >= 4:
                return ' '.join(num2words(int(d), lang='es') for d in s)
            try: return num2words(int(s), lang='es')
            except: return s
        return re.sub(r'\b\d+\b', convert, text)

    def _spanish_post_processing(self, text):
        replacements = [
            (r'\buno terabyte\b', 'un terabyte'), (r'\buno gigabyte\b', 'un gigabyte'),
            (r'\buno monitor\b', 'un monitor'), (r'\buno soporte\b', 'un soporte'),
            (r'\buno escritorio\b', 'un escritorio'), (r'\buno pedido\b', 'un pedido')
        ]
        for p, r in replacements: text = re.sub(p, r, text)
        return text
