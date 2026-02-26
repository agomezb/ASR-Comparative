"""
Análisis y visualización de resultados BERTScore.
Responsabilidad: agregaciones por provider y presentación en tablas y gráficos.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import itertools
from IPython.display import display, HTML

class BERTScoreAnalyzer:
    """
    Responsable de agregar los datos de BERTScore y calcular métricas por grupo.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def bootstrap_ci(self, series: pd.Series, n_bootstraps: int = 1000, confidence_level: float = 0.95, random_state: int = 42) -> tuple:
        """
        Calcula el intervalo de confianza de la media usando bootstrap.
        """
        if series.empty or len(series) < 2:
            return np.nan, np.nan
        
        rng = np.random.default_rng(random_state)
        n = len(series)
        
        # Vectorized bootstrap
        values = series.values
        indices = rng.integers(0, n, (n_bootstraps, n))
        resampled_values = values[indices]
        means = np.mean(resampled_values, axis=1)
        
        alpha = (1 - confidence_level) / 2
        lower = np.percentile(means, alpha * 100)
        upper = np.percentile(means, (1 - alpha) * 100)
        return lower, upper

    def calculate_stats_by_provider(self) -> pd.DataFrame:
        """
        Calcula media e IC 95% de BERTScore F1 por proveedor.
        """
        providers = sorted(self.df['provider'].unique())
        results = []
        for provider in providers:
            subset = self.df[self.df['provider'] == provider]['bertscore_f1']
            mean_f1 = subset.mean()
            ci_lower, ci_upper = self.bootstrap_ci(subset)
            results.append({
                'provider': provider,
                'mean_f1': mean_f1,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'count': len(subset)
            })
        return pd.DataFrame(results)

    def calculate_stats_by_provider_and_snr(self) -> pd.DataFrame:
        """
        Calcula media e IC 95% de BERTScore F1 por proveedor y nivel de ruido (SNR).
        """
        # Agrupar por provider y snr
        groups = self.df.groupby(['provider', 'snr'])
        results = []
        
        for (provider, snr), group in groups:
            subset = group['bertscore_f1']
            mean_f1 = subset.mean()
            ci_lower, ci_upper = self.bootstrap_ci(subset)
            results.append({
                'provider': provider,
                'snr': snr,
                'mean_f1': mean_f1,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'count': len(subset)
            })
        return pd.DataFrame(results)

    def _run_statistical_tests(self, df_target: pd.DataFrame) -> tuple:
        """
        Lógica interna para ejecutar Friedman y Wilcoxon sobre un DataFrame dado.
        """
        # Crear ID único para pivotar
        df_target['unique_id'] = (
            df_target['person'].astype(str) + '_' + 
            df_target['audio'].astype(str) + '_' + 
            df_target['noise'].astype(str) + '_' + 
            df_target['snr'].astype(str)
        )
        
        # Pivotar: filas=muestras, columnas=proveedores
        pivot_bert = df_target.pivot(index='unique_id', columns='provider', values='bertscore_f1').dropna()
        
        if pivot_bert.empty:
            print("No hay datos suficientes para análisis estadístico.")
            return (None, None, None, None), None
            
        # Test de Friedman
        stat, p_value = stats.friedmanchisquare(*[pivot_bert[col] for col in pivot_bert.columns])
        
        # W de Kendall
        n_blocks = pivot_bert.shape[0]
        k_groups = pivot_bert.shape[1]
        kendall_w = stat / (n_blocks * (k_groups - 1)) if (n_blocks > 0 and k_groups > 1) else None
        friedman_result = (stat, p_value, kendall_w, n_blocks)
        
        wilcoxon_df = None
        if p_value < 0.05:
            comparisons = []
            p_values = []
            w_statistics = []
            effect_sizes = []
            pairs = list(itertools.combinations(pivot_bert.columns, 2))
            n_comparisons = len(pairs)
            
            for p1, p2 in pairs:
                # Wilcoxon test
                try:
                    stat_w, p_w = stats.wilcoxon(pivot_bert[p1], pivot_bert[p2])
                except ValueError:
                    stat_w, p_w = 0, 1.0
                    
                comparisons.append(f"{p1} vs {p2}")
                p_values.append(p_w)
                w_statistics.append(stat_w)
                
                # Tamaño del efecto (r = Z / sqrt(N))
                n = len(pivot_bert)
                mu = n * (n + 1) / 4
                se = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
                
                if se > 0:
                    z = (stat_w - mu) / se
                    r = abs(z) / np.sqrt(n)
                else:
                    r = 0.0
                effect_sizes.append(r)
            
            # Ajuste Holm-Bonferroni
            sorted_indices = np.argsort(p_values)
            sorted_p_values = np.array(p_values)[sorted_indices]
            p_adjusted_sorted = []
            for i, p in enumerate(sorted_p_values):
                m_i = n_comparisons - i
                p_adj = min(1.0, p * m_i)
                if i > 0:
                    p_adj = max(p_adj, p_adjusted_sorted[-1])
                p_adjusted_sorted.append(p_adj)
            p_adjusted = [0.0] * n_comparisons
            for i, idx in enumerate(sorted_indices):
                p_adjusted[idx] = p_adjusted_sorted[i]
            reject = [p < 0.05 for p in p_adjusted]

            wilcoxon_df = pd.DataFrame({
                'Comparación': comparisons,
                'Estadístico W': w_statistics,
                'p-value original': p_values,
                'p-value adj (Holm-Bonferroni)': p_adjusted,
                'Significativo': reject,
                'Tamaño del Efecto (r)': effect_sizes
            })
            
        return friedman_result, wilcoxon_df

    def perform_statistical_tests(self) -> tuple:
        """
        Realiza pruebas estadísticas (Friedman y Wilcoxon post-hoc) para comparar ASR sobre todo el dataset.
        """
        return self._run_statistical_tests(self.df.copy())

    def perform_statistical_tests_by_snr(self, snr_level: str) -> tuple:
        """
        Realiza pruebas estadísticas filtrando por un nivel de SNR específico.
        """
        df_filtered = self.df[self.df['snr'] == snr_level].copy()
        if df_filtered.empty:
             print(f"No hay datos para el nivel de ruido: {snr_level}")
             return (None, None, None, None), None
        return self._run_statistical_tests(df_filtered)

class BERTScoreVisualizer:
    """
    Responsable de mostrar tablas y gráficos de BERTScore.
    """
    @staticmethod
    def plot_bertscore_by_provider(stats_df: pd.DataFrame, title: str = "BERTScore F1 por ASR con intervalo de confianza (95%)"):
        if stats_df.empty:
            print("No hay datos para graficar.")
            return

        providers = stats_df['provider'].unique()
        
        colors = {
            'google': '#4285F4',
            'azure': '#0078D4',
            'amazon': '#FF9900',
            'whisper': '#34A853'
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(providers))
        means = stats_df['mean_f1'].values
        yerr_lower = means - stats_df['ci_lower'].values
        yerr_upper = stats_df['ci_upper'].values - means
        
        # Ensure non-negative error bars
        yerr_lower = np.maximum(yerr_lower, 0)
        yerr_upper = np.maximum(yerr_upper, 0)
        
        bar_colors = [colors.get(p, 'gray') for p in providers]

        ax.bar(x, means, yerr=[yerr_lower, yerr_upper], capsize=10,
               color=bar_colors, alpha=0.9, edgecolor='white', width=0.6)
        
        ax.set_ylabel('BERTScore F1', fontsize=12, fontweight='bold')
        ax.set_xlabel('ASR', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(p).capitalize() for p in providers], fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        for i, (m, eu) in enumerate(zip(means, yerr_upper)):
            ax.text(i, m + eu + 0.02, f'{m:.3f}', ha='center', fontweight='bold', fontsize=10)
            
        plt.tight_layout()
        plt.show()

    @staticmethod
    def display_summary_table(stats_df: pd.DataFrame):
        display_df = stats_df.copy()
        display_df['ASR'] = display_df['provider'].str.capitalize()
        display_df = display_df[['ASR', 'mean_f1', 'ci_lower', 'ci_upper']]
        display_df.columns = ['ASR', 'BERTScore F1', 'IC 95% Inf', 'IC 95% Sup']
        
        print("Tabla: BERTScore F1 por ASR con intervalos de confianza (95%)")
        display(display_df.style.format({
            'BERTScore F1': '{:.4f}',
            'IC 95% Inf': '{:.4f}',
            'IC 95% Sup': '{:.4f}'
        }).hide(axis='index'))

    @staticmethod
    def plot_bertscore_boxplot(df: pd.DataFrame, title: str = "Distribución de BERTScore F1 por ASR"):
        """
        Genera un gráfico de cajas (boxplot) para la distribución de BERTScore F1 por proveedor.
        """
        if df.empty:
            print("No hay datos para graficar.")
            return

        providers = sorted(df['provider'].unique())
        data_by_provider = [df[df['provider'] == p]['bertscore_f1'].values for p in providers]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bp = ax.boxplot(data_by_provider, labels=[str(p).capitalize() for p in providers], patch_artist=True)
        
        colors = {
            'google': '#4285F4',
            'azure': '#0078D4',
            'amazon': '#FF9900',
            'whisper': '#34A853'
        }
        
        for patch, label in zip(bp['boxes'], providers):
            patch.set_facecolor(colors.get(label, 'gray'))
            patch.set_alpha(0.7)
            
        ax.set_ylabel('BERTScore F1', fontsize=12, fontweight='bold')
        ax.set_xlabel('ASR', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_bertscore_by_snr(stats_df: pd.DataFrame, title: str = "BERTScore F1 por Nivel de Ruido y ASR"):
        """
        Genera un gráfico de líneas de BERTScore por SNR y proveedor.
        """
        if stats_df.empty:
            print("No hay datos para graficar.")
            return

        # Ordenar niveles de SNR
        snr_order = ['clean', '10dB', '5dB', '0dB']
        # Filtrar solo los que existen en los datos
        snr_levels = [s for s in snr_order if s in stats_df['snr'].unique()]
        
        providers = sorted(stats_df['provider'].unique())
        
        colors = {
            'google': '#4285F4',
            'azure': '#0078D4',
            'amazon': '#FF9900',
            'whisper': '#34A853'
        }
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        x = np.arange(len(snr_levels))
        
        # Variables para calcular límites del eje Y
        min_y = 1.0
        max_y = 0.0
        
        for provider in providers:
            provider_data = stats_df[stats_df['provider'] == provider]
            # Reindexar para asegurar el orden correcto de SNR
            provider_data = provider_data.set_index('snr').reindex(snr_levels).reset_index()
            
            means = provider_data['mean_f1'].values
            ci_lower = provider_data['ci_lower'].values
            ci_upper = provider_data['ci_upper'].values
            
            # Actualizar límites
            valid_lower = ci_lower[~np.isnan(ci_lower)]
            valid_upper = ci_upper[~np.isnan(ci_upper)]
            if len(valid_lower) > 0:
                min_y = min(min_y, np.min(valid_lower))
            if len(valid_upper) > 0:
                max_y = max(max_y, np.max(valid_upper))
            
            # Calcular errores asimétricos
            yerr_lower = np.maximum(means - ci_lower, 0)
            yerr_upper = np.maximum(ci_upper - means, 0)
            
            # Manejar NaNs para que no rompan el gráfico si faltan datos intermedios (opcional, pero errorbar maneja NaNs ignorándolos)
            
            ax.errorbar(x, means, yerr=[yerr_lower, yerr_upper], label=str(provider).capitalize(),
                        color=colors.get(provider, 'gray'), capsize=4, marker='o', linewidth=2, markersize=6)
            
        ax.set_ylabel('BERTScore F1', fontsize=12, fontweight='bold')
        ax.set_xlabel('Nivel de Ruido (SNR)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(snr_levels, fontsize=11)
        ax.legend(title='ASR', fontsize=11, loc='lower left')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Ajustar escala Y dinámicamente con un margen
        if min_y < max_y:
            margin = (max_y - min_y) * 0.1
            ax.set_ylim(max(0, min_y - margin), min(1.02, max_y + margin))
        else:
            ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def display_summary_table_by_snr(stats_df: pd.DataFrame):
        """
        Muestra una tabla resumen de BERTScore por SNR y proveedor en formato ancho.
        Columnas: ASR, Base, 10 dB, 5 dB, 0 dB, Degradación.
        Valores: Mean [CI Lower, CI Upper]
        """
        if stats_df.empty:
            print("No hay datos para mostrar.")
            return

        # Preparar datos
        data = []
        providers = sorted(stats_df['provider'].unique())
        
        for provider in providers:
            p_data = stats_df[stats_df['provider'] == provider]
            row = {'ASR': str(provider).capitalize()}
            
            # Mapeo de nombres de SNR
            snr_map = {
                'clean': 'Base',
                '10dB': '10 dB',
                '5dB': '5 dB',
                '0dB': '0 dB'
            }
            
            base_score = None
            zero_db_score = None
            
            for _, item in p_data.iterrows():
                snr_key = snr_map.get(item['snr'], item['snr'])
                formatted_val = f"{item['mean_f1']:.4f} [{item['ci_lower']:.4f}, {item['ci_upper']:.4f}]"
                row[snr_key] = formatted_val
                
                if item['snr'] == 'clean':
                    base_score = item['mean_f1']
                if item['snr'] == '0dB':
                    zero_db_score = item['mean_f1']
            
            # Calcular Degradación (Base - 0 dB)
            if base_score is not None and zero_db_score is not None:
                degradation = base_score - zero_db_score
                row['Degradación'] = f"{degradation:.4f}"
            else:
                row['Degradación'] = "-"
                
            data.append(row)
            
        display_df = pd.DataFrame(data)
        
        # Asegurar orden de columnas
        cols = ['ASR', 'Base', '10 dB', '5 dB', '0 dB', 'Degradación']
        # Filtrar solo columnas que existen (por si falta algún SNR)
        cols = [c for c in cols if c in display_df.columns]
        display_df = display_df[cols]
        
        print("\nTabla: BERTScore F1 por Nivel de Ruido y ASR")
        display(display_df.style.hide(axis='index').set_properties(**{'text-align': 'center'}))

    @staticmethod
    def display_detailed_statistics(df: pd.DataFrame, title: str = "Estadísticas Descriptivas de BERTScore F1 por ASR"):
        """
        Muestra una tabla con: ASR, Media, Mediana, Mínimo, Máximo, Desv. Est., Q1, Q3, IQR, Muestras.
        """
        if df.empty:
            print("No hay datos para mostrar.")
            return

        agg = df.groupby('provider')['bertscore_f1'].agg([
            ('Media', 'mean'),
            ('Mediana', 'median'),
            ('Mínimo', 'min'),
            ('Máximo', 'max'),
            ('Desv. Est.', 'std'),
            ('Muestras', 'count'),
        ])
        q1 = df.groupby('provider')['bertscore_f1'].quantile(0.25)
        q3 = df.groupby('provider')['bertscore_f1'].quantile(0.75)
        agg.insert(agg.columns.get_loc('Desv. Est.') + 1, 'Q1', q1)
        agg.insert(agg.columns.get_loc('Q1') + 1, 'Q3', q3)
        agg.insert(agg.columns.get_loc('Q3') + 1, 'IQR', q3 - q1)
        agg = agg.reset_index()
        agg = agg.rename(columns={'provider': 'ASR'})
        agg['ASR'] = agg['ASR'].str.capitalize()
        # Orden de columnas: ASR, Media, Mediana, Mínimo, Máximo, Desv. Est., Q1, Q3, IQR, Muestras
        stats = agg[['ASR', 'Media', 'Mediana', 'Mínimo', 'Máximo', 'Desv. Est.', 'Q1', 'Q3', 'IQR', 'Muestras']]

        print(f"\n{title}")
        display(stats.style.format({
            'Media': '{:.4f}',
            'Mediana': '{:.4f}',
            'Mínimo': '{:.4f}',
            'Máximo': '{:.4f}',
            'Desv. Est.': '{:.4f}',
            'Q1': '{:.4f}',
            'Q3': '{:.4f}',
            'IQR': '{:.4f}',
            'Muestras': '{:.0f}',
        }).hide(axis='index').set_properties(**{'text-align': 'center'}))

    @staticmethod
    def display_statistical_results(friedman_result, wilcoxon_df, title: str = "Análisis Estadístico de Significancia (BERTScore F1)") -> None:
        """
        Muestra los resultados de Friedman y post-hoc (Wilcoxon + Holm-Bonferroni).
        """
        stat, p_value = friedman_result[0], friedman_result[1]
        kendall_w = friedman_result[2] if len(friedman_result) > 2 else None
        n_blocks = friedman_result[3] if len(friedman_result) > 3 else None
        
        if stat is None:
            print(f"\n{title}")
            print("No se pudieron realizar las pruebas estadísticas.")
            return

        print(f"\n{title}")
        if n_blocks is not None:
            print(f"N (muestras/bloques) = {n_blocks}")
        print(f"Test de Friedman: Estadístico={stat:.4f}, p-value={p_value:.4e}", end="")
        if kendall_w is not None:
            print(f", W de Kendall={kendall_w:.4f}")
        else:
            print()

        if p_value < 0.05 and wilcoxon_df is not None:
            print(">> Diferencias significativas encontradas. Realizando post-hoc (Wilcoxon + Holm-Bonferroni)...")
            fmt = {
                'Estadístico W': '{:.0f}',
                'p-value original': '{:.4e}',
                'p-value adj (Holm-Bonferroni)': '{:.4e}',
                'Tamaño del Efecto (r)': '{:.4f}'
            }
            fmt = {k: v for k, v in fmt.items() if k in wilcoxon_df.columns}
            display(wilcoxon_df.style.format(fmt).hide(axis='index').set_properties(**{
                'text-align': 'center',
                'padding': '8px'
            }).set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'center'), ('font-weight', 'bold')]}
            ]))
        else:
            print(">> No se encontraron diferencias significativas.")
