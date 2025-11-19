#!/usr/bin/env python3
"""
Script de normalización de texto para evaluación de sistemas ASR.
Aplica reglas estrictas de normalización para cálculo de WER en español.
"""

import re
from typing import Dict, Optional
from pathlib import Path

import pandas as pd
import typer
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
            "1tb": "un terabyte",
            "2tb": "dos terabyte",
            "3tb": "tres terabyte",
            "4tb": "cuatro terabyte",
            "tb": "terabyte",
            # Procesadores
            "i7": "i siete",
            "i5": "i cinco",
            "i3": "i tres",
            # Formatos
            "a4": "a cuatro",
            # Modelos
            "xg": "equis ge",
            # Códigos/Facturas
            "f a": "efe a",
            "fa": "efe a",
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
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        
        # Insertar espacio entre número y letra
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        
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
            
            # Si es una secuencia larga (más de 4 dígitos), tratar como código
            # y convertir dígito por dígito
            if len(number_str) > 4:
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


class CSVNormalizer:
    """
    Clase responsable de procesar archivos CSV y normalizar columnas de texto.
    
    Sigue el principio de Single Responsibility: solo maneja la lógica de CSV.
    """
    
    def __init__(self, normalizer: TextNormalizer):
        """
        Inicializa el procesador de CSV.
        
        Args:
            normalizer: Instancia de TextNormalizer para normalizar texto.
        """
        self.normalizer = normalizer
    
    def process_csv(
        self,
        input_path: Path,
        output_path: Path,
        text_column: str = "text"
    ) -> None:
        """
        Procesa un archivo CSV normalizando la columna de texto especificada.
        
        Args:
            input_path: Ruta al archivo CSV de entrada.
            output_path: Ruta al archivo CSV de salida.
            text_column: Nombre de la columna a normalizar.
            
        Raises:
            FileNotFoundError: Si el archivo de entrada no existe.
            KeyError: Si la columna especificada no existe en el CSV.
        """
        # Leer CSV
        df = pd.read_csv(input_path)
        
        # Verificar que la columna existe
        if text_column not in df.columns:
            raise KeyError(
                f"La columna '{text_column}' no existe en el CSV. "
                f"Columnas disponibles: {list(df.columns)}"
            )
        
        # Normalizar la columna de texto
        df[f"{text_column}_normalized"] = df[text_column].apply(
            self.normalizer.normalize
        )
        
        # Guardar CSV normalizado
        df.to_csv(output_path, index=False)


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


# CLI con Typer
app = typer.Typer(
    help="Herramienta de normalización de texto para evaluación de sistemas ASR."
)


@app.command()
def normalizar(
    input_csv: Path = typer.Argument(
        ...,
        help="Ruta al archivo CSV de entrada.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_csv: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Ruta al archivo CSV de salida. Por defecto: {input}_normalized.csv",
    ),
    text_column: str = typer.Option(
        "text",
        "--column",
        "-c",
        help="Nombre de la columna a normalizar.",
    ),
) -> None:
    """
    Normaliza el texto de un archivo CSV para evaluación ASR.
    
    Aplica las siguientes transformaciones:
    \n
    - Conversión a minúsculas
    \n
    - Eliminación de puntuación
    \n
    - Conversión de números a palabras (español)
    \n
    - Reemplazos de dominio personalizados
    \n
    - Limpieza de espacios
    
    Ejemplo:
    \n
        $ python normalizar_texto.py google_clean.csv
    \n
        $ python normalizar_texto.py google_clean.csv -o google_normalized.csv
    """
    try:
        # Determinar ruta de salida
        if output_csv is None:
            output_csv = input_csv.parent / f"{input_csv.stem}_normalized{input_csv.suffix}"
        
        # Crear normalizador y procesador
        normalizer = TextNormalizer()
        csv_processor = CSVNormalizer(normalizer)
        
        # Procesar archivo
        typer.echo(f"Procesando: {input_csv}")
        csv_processor.process_csv(input_csv, output_csv, text_column)
        
        # Mostrar estadísticas
        df = pd.read_csv(output_csv)
        typer.echo(f"Normalizacion completada exitosamente!")
        typer.echo(f"Filas procesadas: {len(df)}")
        typer.echo(f"Archivo guardado en: {output_csv}")
        
        # Mostrar ejemplo
        if len(df) > 0:
            typer.echo("\nEjemplo de normalizacion:")
            typer.echo(f"  Original:    {df[text_column].iloc[0][:80]}...")
            typer.echo(f"  Normalizado: {df[f'{text_column}_normalized'].iloc[0][:80]}...")
        
    except FileNotFoundError:
        typer.echo(f"Error: El archivo '{input_csv}' no existe.", err=True)
        raise typer.Exit(code=1)
    
    except KeyError as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(code=1)
    
    except Exception as e:
        typer.echo(f"Error inesperado: {str(e)}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

