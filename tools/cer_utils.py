"""
Utilidades para el cálculo de CER (Character Error Rate) por keywords.
Responsabilidad: carga de datos, alineación con jiwer y cálculo de distancia de edición por palabra clave.
"""
import json
import pandas as pd
import jiwer
import Levenshtein


def load_keywords(keywords_path: str) -> dict:
    """
    Carga el archivo JSON de keywords (audio_id -> text + keywords).

    Args:
        keywords_path: Ruta al archivo keywords.json.

    Returns:
        Diccionario con la estructura del JSON.
    """
    with open(keywords_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Carga el dataset normalizado (CSV).

    Args:
        dataset_path: Ruta al CSV (ej. dataset_normalized.csv).

    Returns:
        DataFrame del dataset.
    """
    return pd.read_csv(dataset_path)


def _get_cer_stats_for_row(row: pd.Series, keywords_data: dict) -> list[dict]:
    """
    Calcula las estadísticas de CER (edit_distance, ref_length) para una fila del dataset
    usando alineación jiwer y Levenshtein por cada keyword.

    Args:
        row: Fila del DataFrame (debe tener 'audio', 'text_normalized', 'noise', 'snr', 'provider').
        keywords_data: Diccionario de keywords por audio_id.

    Returns:
        Lista de diccionarios con audio, noise, snr, provider, category, ref_word, hyp_word,
        edit_distance, ref_length. Lista vacía si el audio no está en keywords.
    """
    audio_id = str(row["audio"])
    if audio_id not in keywords_data:
        return []

    ref_text = keywords_data[audio_id]["text"]
    hyp_text = str(row["text_normalized"]) if pd.notna(row["text_normalized"]) else ""

    output = jiwer.process_words(ref_text, hyp_text)
    alignment = output.alignments[0]
    hypothesis_words = output.hypotheses[0]

    stats = []
    for kw in keywords_data[audio_id]["keywords"]:
        ref_idx = kw["word_index"]
        # Support both old 'key' and new 'subcategory'/'category' structure
        subcategory = kw.get("subcategory", kw.get("key"))
        category = kw.get("category", "Otros")
        ref_word = kw["val"]

        found_chunk = None
        for chunk in alignment:
            if chunk.ref_start_idx <= ref_idx < chunk.ref_end_idx:
                found_chunk = chunk
                break

        hyp_word = ""
        if found_chunk:
            if found_chunk.type in ("equal", "substitute"):
                offset = ref_idx - found_chunk.ref_start_idx
                hyp_idx = found_chunk.hyp_start_idx + offset
                if hyp_idx < found_chunk.hyp_end_idx:
                    hyp_word = hypothesis_words[hyp_idx]
            elif found_chunk.type == "delete":
                hyp_word = ""

        dist = Levenshtein.distance(ref_word, hyp_word)
        ref_len = len(ref_word)

        stats.append({
            "audio": row["audio"],
            "noise": row["noise"],
            "snr": row["snr"],
            "provider": row["provider"],
            "category": category, # Now using the Super Category as the main 'category'
            "subcategory": subcategory, # Preserving specific key as 'subcategory'
            "ref_word": ref_word,
            "hyp_word": hyp_word,
            "edit_distance": dist,
            "ref_length": ref_len,
        })

    return stats


def calculate_cer_from_dataframe(
    dataset: pd.DataFrame,
    keywords_path: str = "keywords.json",
) -> pd.DataFrame:
    """
    Calcula el CER por keyword para todo el dataset y devuelve un DataFrame de estadísticas.

    Args:
        dataset: DataFrame con columnas audio, text_normalized, noise, snr, provider.
        keywords_path: Ruta al archivo keywords.json.

    Returns:
        DataFrame con columnas: audio, noise, snr, provider, category, ref_word, hyp_word,
        edit_distance, ref_length (una fila por keyword por fila del dataset).
    """
    keywords_data = load_keywords(keywords_path)
    all_stats = []
    for _, row in dataset.iterrows():
        row_stats = _get_cer_stats_for_row(row, keywords_data)
        all_stats.extend(row_stats)
    return pd.DataFrame(all_stats)
