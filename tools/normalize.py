#!/usr/bin/env python3
"""
Script de normalización de texto para evaluación de sistemas ASR.
Aplica reglas estrictas de normalización para cálculo de WER en español.
"""

import re
from typing import Dict, Optional
from num2words import num2words


class TextNormalizer:
    """
    Clase responsable de normalizar texto en español para evaluación ASR.
    
    Aplica las siguientes transformaciones:
    - Conversión a minúsculas
    - Eliminación de puntuación
    - Conversión de números a palabras
    - Reemplazos de dominio personalizados
    - Limpieza de espacios
    """
    
    def __init__(self, custom_replacements: Optional[Dict[str, str]] = None):
        """
        Inicializa el normalizador con reemplazos personalizados.
        
        Args:
            custom_replacements: Diccionario de reemplazos personalizados.
        """
        self.custom_replacements = custom_replacements or self._get_default_replacements()
    
    @staticmethod
    def _get_default_replacements() -> Dict[str, str]:
        """Retorna el diccionario de reemplazos de dominio por defecto."""
        return {
            # Almacenamiento
            "1 tb": "un terabyte",
            "2 tb": "dos terabyte",
            "3 tb": "tres terabyte",
            "4 tb": "cuatro terabyte",
            "tb": "terabyte",

            # Procesadores
            "i 7": "i siete",
            "i 5": "i cinco",
            "i 3": "i tres",

            # Formatos
            "a 4": "a cuatro",

            # Modelos
            "xg": "equis ge",

            # Códigos/Facturas
            "f a": "efe a",
            "fa": "efe a",

            # Marcas
            "compufacil": "compu facil",
            "compufácil": "compu facil",
            "andinacorp": "andina corp",
            "duradisco": "dura disco"
        }
    
    def normalize(self, text: str) -> str:
        """
        Aplica todas las reglas de normalización al texto.
        
        Args:
            text: Texto original a normalizar.
            
        Returns:
            Texto normalizado.
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Paso 1: Convertir a minúsculas
        text = self._to_lowercase(text)
        
        # Paso 2: Separar letras y números pegados (CRÍTICO para Amazon ASR)
        text = self._separate_letters_and_numbers(text)
        
        # --- NECESARIO PARA WHISPER ---
        # Paso 2.5: Unir números partidos por guiones (85-20 -> 8520)
        # Esto asegura que se detecten como series largas
        text = re.sub(r'(?<=\d)-(?=\d)', '', text)
        # ---------------------------------------

        # Paso 3: Aplicar reemplazos personalizados (antes de eliminar puntuación)
        text = self._apply_custom_replacements(text)
        
        # Paso 4: Convertir números a palabras
        text = self._numbers_to_words(text)
        
        # Paso 5: Eliminar puntuación
        text = self._remove_punctuation(text)
        
        # Paso 6: Post-procesamiento específico de español
        text = self._spanish_post_processing(text)
        
        # Paso 7: Limpiar espacios
        text = self._clean_whitespace(text)
        
        return text
    
    @staticmethod
    def _to_lowercase(text: str) -> str:
        """Convierte el texto a minúsculas."""
        return text.lower()
    
    @staticmethod
    def _separate_letters_and_numbers(text: str) -> str:
        """
        Separa letras y números que están concatenados sin espacios.
        
        Este método soluciona el problema de ASR (especialmente Amazon) que 
        concatena letras y números, generando cadenas como:
        - FA409516 -> FA 409516
        - RU09220783 -> RU 09220783
        - i73 -> i 73
        
        Args:
            text: Texto con posibles concatenaciones.
            
        Returns:
            Texto con letras y números separados por espacios.
        """
        # Insertar espacio entre letra y número
        text = re.sub(r'([a-zA-ZáéíóúñÁÉÍÓÚÑ])(\d)', r'\1 \2', text)
        
        # Insertar espacio entre número y letra
        text = re.sub(r'(\d)([a-zA-ZáéíóúñÁÉÍÓÚÑ])', r'\1 \2', text)
        
        return text
    
    def _apply_custom_replacements(self, text: str) -> str:
        """
        Aplica reemplazos personalizados de dominio.
        
        Los reemplazos se aplican con word boundaries para evitar
        reemplazos parciales no deseados.
        """
        for original, replacement in self.custom_replacements.items():
            # Usar word boundary para reemplazos exactos
            pattern = r'\b' + re.escape(original) + r'\b'
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _numbers_to_words(self, text: str) -> str:
        """
        Convierte números a palabras en español.
        
        Maneja:
        - Números individuales (5 -> cinco)
        - Números largos (secuencias de más de 4 dígitos se tratan dígito por dígito)
        """
        def convert_number(match):
            number_str = match.group(0)
            
            # Si es una secuencia larga (4 o más dígitos), tratar como código
            # y convertir dígito por dígito
            if len(number_str) >= 4:
                return ' '.join(num2words(int(digit), lang='es') for digit in number_str)
            
            # Números normales: convertir directamente
            try:
                number = int(number_str)
                return num2words(number, lang='es')
            except (ValueError, OverflowError):
                return number_str
        
        # Buscar secuencias de dígitos
        text = re.sub(r'\b\d+\b', convert_number, text)
        
        return text
    
    @staticmethod
    def _remove_punctuation(text: str) -> str:
        """
        Elimina todos los signos de puntuación.
        
        Mantiene solo letras, números (ya convertidos a palabras) y espacios.
        """
        # Eliminar todos los caracteres excepto letras, números y espacios
        text = re.sub(r'[^\w\s]', ' ', text)
        # Eliminar guiones bajos que quedan del \w
        text = re.sub(r'_', ' ', text)
        
        return text
    
    @staticmethod
    def _spanish_post_processing(text: str) -> str:
        """
        Aplica reglas específicas de español después de la normalización.
        
        Principalmente convierte "uno" a "un" cuando es un artículo.
        Por ejemplo: "uno terabyte" -> "un terabyte"
        """
        # Convertir "uno" a "un" antes de sustantivos comunes
        text = re.sub(r'\buno terabyte\b', 'un terabyte', text)
        text = re.sub(r'\buno gigabyte\b', 'un gigabyte', text)
        text = re.sub(r'\buno megabyte\b', 'un megabyte', text)
        
        return text
    
    @staticmethod
    def _clean_whitespace(text: str) -> str:
        """Limpia espacios múltiples y espacios al inicio/final."""
        # Reemplazar múltiples espacios por uno solo
        text = re.sub(r'\s+', ' ', text)
        # Eliminar espacios al inicio y final
        text = text.strip()
        
        return text


def normalize_text(text: str) -> str:
    """
    Función wrapper para normalizar texto desde otros scripts.
    
    Args:
        text: Texto a normalizar.
        
    Returns:
        Texto normalizado.
    """
    normalizer = TextNormalizer()
    return normalizer.normalize(text)
