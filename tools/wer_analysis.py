import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import itertools

class WERAnalyzer:
    """
    Clase responsable de analizar los datos de WER, incluyendo cálculos estadísticos
    y bootstrapping para intervalos de confianza.
    """
    def __init__(self, df_wer):
        self.df_wer = df_wer.copy()

    def bootstrap_wer_ci(self, df_group, n_bootstraps=1000, confidence_level=0.95):
        """
        Calcula el intervalo de confianza del WER usando bootstrap sobre un grupo de datos.
        
        Args:
            df_group (pd.DataFrame): DataFrame con columnas 'errors' y 'reference_words'.
            n_bootstraps (int): Número de iteraciones para el bootstrap.
            confidence_level (float): Nivel de confianza (0.95 por defecto).
            
        Returns:
            tuple: (limite_inferior, limite_superior)
        """
        wer_values = []
        
        # Validación de columnas requeridas
        if 'errors' not in df_group.columns or 'reference_words' not in df_group.columns:
             # Fallback: si solo tenemos 'wer' pre-calculado por fila (menos preciso para WER global)
             if 'wer' in df_group.columns:
                 data = df_group['wer'].values
                 for _ in range(n_bootstraps):
                     sample = np.random.choice(data, size=len(data), replace=True)
                     wer_values.append(sample.mean())
             else:
                 raise KeyError("El DataFrame debe contener columnas 'errors' y 'reference_words' o 'wer'.")
        else:
            # Método correcto: Suma de errores / Suma de palabras
            errors = df_group['errors'].values
            words = df_group['reference_words'].values
            n_samples = len(df_group)
            rng = np.random.default_rng(42)
            
            for _ in range(n_bootstraps):
                indices = rng.integers(0, n_samples, n_samples)
                total_errors = errors[indices].sum()
                total_words = words[indices].sum()
                wer = total_errors / total_words if total_words > 0 else 0.0
                wer_values.append(wer)
            
        alpha = (1 - confidence_level) / 2
        lower = np.percentile(wer_values, alpha * 100)
        upper = np.percentile(wer_values, (1 - alpha) * 100)
        return lower, upper

    def calculate_wer_by_provider_snr(self):
        """
        Agrupa los datos por ASR y nivel de ruido (SNR), calculando el WER promedio
        y sus intervalos de confianza.
        
        Returns:
            pd.DataFrame: DataFrame con columnas [provider, snr, wer, ci_lower, ci_upper]
        """
        results = []
        
        if 'snr' not in self.df_wer.columns or 'provider' not in self.df_wer.columns:
             raise ValueError("El DataFrame debe contener columnas 'provider' y 'snr'.")

        for (provider, snr), group in self.df_wer.groupby(['provider', 'snr']):
            # Calcular Estimación Puntual (WER Global para este grupo)
            if 'errors' in group.columns and 'reference_words' in group.columns:
                total_errors = group['errors'].sum()
                total_words = group['reference_words'].sum()
                wer = total_errors / total_words if total_words > 0 else 0.0
            else:
                wer = group['wer'].mean()
            
            # Calcular Intervalos de Confianza
            lower, upper = self.bootstrap_wer_ci(group)
            
            results.append({
                'provider': provider,
                'snr': snr,
                'wer': wer,
                'ci_lower': lower,
                'ci_upper': upper,
                'count': len(group)
            })
        
        return pd.DataFrame(results)

    def calculate_global_wer(self):
        """
        Calcula el WER global por ASR (sin desglosar por SNR).
        
        Returns:
            pd.DataFrame: DataFrame con columnas [provider, wer_global, ci_lower, ci_upper]
        """
        if 'provider' not in self.df_wer.columns:
            raise ValueError("El DataFrame debe contener la columna 'provider'.")

        # Agrupar por ASR y calcular totales
        if 'errors' in self.df_wer.columns and 'reference_words' in self.df_wer.columns:
            summary_df = self.df_wer.groupby('provider').agg(
                total_errors=('errors', 'sum'),
                total_words=('reference_words', 'sum')
            ).reset_index()
            summary_df['wer_global'] = summary_df['total_errors'] / summary_df['total_words']
        else:
            # Fallback a media simple si no hay desglose de errores
            summary_df = self.df_wer.groupby('provider')['wer'].mean().reset_index(name='wer_global')

        # Calcular CIs
        ci_results = []
        for provider, group in self.df_wer.groupby('provider'):
            lower, upper = self.bootstrap_wer_ci(group)
            ci_results.append({'provider': provider, 'ci_lower': lower, 'ci_upper': upper})
        
        ci_df = pd.DataFrame(ci_results)
        summary_df = summary_df.merge(ci_df, on='provider')
        
        return summary_df.sort_values('wer_global')

    def filter_by_snr(self, snr_level):
        """
        Filtra el DataFrame por un nivel de SNR específico.
        
        Args:
            snr_level (str): Nivel de SNR a filtrar (e.g., '0dB', 'clean').
            
        Returns:
            pd.DataFrame: DataFrame filtrado.
        """
        if 'snr' not in self.df_wer.columns:
            raise ValueError("El DataFrame debe contener la columna 'snr'.")
        return self.df_wer[self.df_wer['snr'] == snr_level].copy()

    def perform_statistical_tests(self, df=None):
        """
        Realiza pruebas estadísticas (Friedman y Wilcoxon post-hoc) para comparar ASR.
        
        Args:
            df (pd.DataFrame, optional): DataFrame sobre el cual realizar las pruebas. 
                                         Si es None, usa self.df_wer.
        
        Returns:
            tuple: (friedman_result, wilcoxon_results_df)
                   friedman_result: (statistic, p_value)
                   wilcoxon_results_df: DataFrame con resultados post-hoc o None si Friedman no es significativo.
        """
        target_df = df if df is not None else self.df_wer
        
        # Crear ID único para pivotar: person + audio + noise + snr
        # Asumiendo que estas columnas existen y forman una clave única
        required_cols = ['person', 'audio', 'noise', 'snr', 'provider', 'wer']
        if not all(col in target_df.columns for col in required_cols):
            print("Advertencia: Faltan columnas para crear ID único. Se intentará usar índice existente si es apropiado, o se omitirá el análisis.")
            return (None, None), None

        df_stats = target_df.copy()
        df_stats['unique_id'] = df_stats['person'].astype(str) + '_' + df_stats['audio'].astype(str) + '_' + df_stats['noise'].astype(str) + '_' + df_stats['snr'].astype(str)
        
        pivot_wer = df_stats.pivot(index='unique_id', columns='provider', values='wer').dropna()
        
        if pivot_wer.empty:
            print("Advertencia: No hay datos suficientes para el análisis estadístico después de pivotar.")
            return (None, None), None

        # Test de Friedman
        stat, p_value = stats.friedmanchisquare(*[pivot_wer[col] for col in pivot_wer.columns])
        friedman_result = (stat, p_value)
        
        wilcoxon_df = None
        if p_value < 0.05:
            comparisons = []
            p_values = []
            effect_sizes = []
            pairs = list(itertools.combinations(pivot_wer.columns, 2))
            n_comparisons = len(pairs)
            
            for p1, p2 in pairs:
                # Wilcoxon test
                try:
                    stat_w, p_w = stats.wilcoxon(pivot_wer[p1], pivot_wer[p2])
                except ValueError:
                    # Handle case where all differences are zero
                    stat_w, p_w = 0, 1.0
                
                comparisons.append(f"{p1} vs {p2}")
                p_values.append(p_w)
                
                # Effect size (r = Z / sqrt(N))
                n = len(pivot_wer)
                
                # Z-score approximation
                mu = n * (n + 1) / 4
                se = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
                
                if se > 0:
                    z = (stat_w - mu) / se
                    r = abs(z) / np.sqrt(n)
                else:
                    r = 0.0
                
                effect_sizes.append(r)
            
            # Holm-Bonferroni correction
            # 1. Sort p-values
            sorted_indices = np.argsort(p_values)
            sorted_p_values = np.array(p_values)[sorted_indices]
            
            # 2. Calculate adjusted p-values
            # p_adj = min(1, p * (m - rank + 1))
            # Ensure monotonicity: p_adj[i] = max(p_adj[i], p_adj[i-1])
            p_adjusted_sorted = []
            for i, p in enumerate(sorted_p_values):
                m_i = n_comparisons - i
                p_adj = min(1.0, p * m_i)
                if i > 0:
                    p_adj = max(p_adj, p_adjusted_sorted[-1])
                p_adjusted_sorted.append(p_adj)
            
            # 3. Restore original order
            p_adjusted = [0.0] * n_comparisons
            for i, idx in enumerate(sorted_indices):
                p_adjusted[idx] = p_adjusted_sorted[i]
            
            reject = [p < 0.05 for p in p_adjusted]
            
            wilcoxon_df = pd.DataFrame({
                'Comparación': comparisons,
                'p-value original': p_values,
                'p-value adj (Holm-Bonferroni)': p_adjusted,
                'Significativo': reject,
                'Tamaño del Efecto (r)': effect_sizes
            })
            
        return friedman_result, wilcoxon_df

class WERVisualizer:
    """
    Clase responsable de la visualización de los resultados de WER.
    """
    def __init__(self):
        self.colors = {
            'google': '#4285F4',
            'azure': '#0078D4',
            'amazon': '#FF9900',
            'whisper': '#34A853'
        }
        self.snr_order = ['clean', '10dB', '5dB', '0dB']

    def plot_wer_by_snr(self, df_results, title='WER por Nivel de Ruido y ASR'):
        """
        Genera un gráfico de líneas con barras de error para el WER por SNR.
        """
        plt.figure(figsize=(12, 7))
        
        providers = df_results['provider'].unique()
        
        for provider in providers:
            p_data = df_results[df_results['provider'] == provider]
            
            # Asegurar orden y continuidad
            p_full = pd.DataFrame({'snr': self.snr_order})
            p_full = p_full.merge(p_data, on='snr', how='left')
            
            # Calcular errores asimétricos para las barras
            if 'ci_lower' in p_full.columns:
                yerr_lower = p_full['wer'] - p_full['ci_lower']
                yerr_upper = p_full['ci_upper'] - p_full['wer']
                # Reemplazar NaNs con 0 para evitar errores en el gráfico
                yerr_lower = yerr_lower.fillna(0)
                yerr_upper = yerr_upper.fillna(0)
                yerr = [yerr_lower, yerr_upper]
            else:
                yerr = None

            plt.errorbar(p_full['snr'], 
                         p_full['wer'], 
                         yerr=yerr,
                         marker='o', 
                         linewidth=2.5, 
                         markersize=8,
                         capsize=5,
                         capthick=2,
                         elinewidth=2,
                         label=provider.capitalize(),
                         color=self.colors.get(provider, None),
                         linestyle='-')

        plt.xlabel('Nivel de Ruido (SNR)', fontsize=13, fontweight='bold')
        plt.ylabel('WER Promedio (con IC 95%)', fontsize=13, fontweight='bold')
        plt.legend(title='ASR', fontsize=11, title_fontsize=12, loc='best', framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()

    def display_comparative_table(self, df_results):
        """
        Muestra una tabla comparativa formateada con degradación.
        """
        # Pivotar tabla
        pivot_df = df_results.pivot(index='provider', columns='snr', values='wer')
        
        # Reordenar columnas
        cols = [c for c in self.snr_order if c in pivot_df.columns]
        pivot_df = pivot_df[cols]
        
        # Calcular degradación
        if 'clean' in pivot_df.columns and '0dB' in pivot_df.columns:
            pivot_df['Degradación (Delta)'] = (pivot_df['0dB'] - pivot_df['clean']) / pivot_df['clean']
            
        pivot_df = pivot_df.reset_index().rename(columns={'provider': 'ASR'})
        
        print("\nTabla Comparativa de WER por Escenario y Degradación:")
        return pivot_df.style.format({
            'clean': '{:.2%}',
            '10dB': '{:.2%}',
            '5dB': '{:.2%}',
            '0dB': '{:.2%}',
            'Degradación (Delta)': '{:+.2%}'
        }).hide(axis='index').set_properties(**{
            'text-align': 'center',
            'padding': '8px'
        }).set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center'), ('font-weight', 'bold')]}
        ])

    def plot_global_wer(self, summary_df):
        """
        Genera un gráfico de barras para el WER global con intervalos de confianza.
        """
        plt.figure(figsize=(10, 6))
        
        yerr = [
            summary_df['wer_global'] - summary_df['ci_lower'],
            summary_df['ci_upper'] - summary_df['wer_global']
        ]
        
        # Asegurar que los errores no sean negativos (por redondeo o bootstrap)
        yerr[0] = np.maximum(yerr[0], 0)
        yerr[1] = np.maximum(yerr[1], 0)

        # Usar paleta consistente si es posible, o viridis
        palette = [self.colors.get(p, '#333333') for p in summary_df['provider']]
        
        bars = plt.bar(summary_df['provider'], summary_df['wer_global'], capsize=10, 
                       yerr=yerr, color=palette)
        
        plt.ylabel('WER (Word Error Rate)', fontsize=12)
        plt.xlabel('ASR', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                     f'{height:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.show()

    def display_global_wer_table(self, summary_df):
        """
        Muestra la tabla de WER global formateada.
        """
        # Renombrar para visualización
        display_df = summary_df.rename(columns={
            'provider': 'ASR',
            'wer_global': 'WER Global',
            'ci_lower': 'IC 95% Inf',
            'ci_upper': 'IC 95% Sup'
        })
        
        print("Tabla Comparativa: Rendimiento Global con Intervalos de Confianza")
        return display_df[['ASR', 'WER Global', 'IC 95% Inf', 'IC 95% Sup']].style.format({
            'WER Global': '{:.2%}',
            'IC 95% Inf': '{:.2%}',
            'IC 95% Sup': '{:.2%}'
        }).hide(axis='index')

    def display_statistical_results(self, friedman_result, wilcoxon_df, title="Análisis Estadístico de Significancia"):
        """
        Muestra los resultados de las pruebas estadísticas.
        """
        stat, p_value = friedman_result
        if stat is None:
            print(f"\n {title} ")
            print("No se pudieron realizar las pruebas estadísticas (datos insuficientes o estructura incorrecta).")
            return

        print(f"\n {title} ")
        print(f"Test de Friedman: Estadístico={stat:.4f}, p-value={p_value:.4e}")
        
        if p_value < 0.05 and wilcoxon_df is not None:
            print(">> Diferencias significativas encontradas. Realizando post-hoc (Wilcoxon + Holm-Bonferroni)...")
            display(wilcoxon_df.style.format({
                'p-value original': '{:.4e}',
                'p-value adj (Holm-Bonferroni)': '{:.4e}',
                'Tamaño del Efecto (r)': '{:.4f}'
            }).hide(axis='index').set_properties(**{
                'text-align': 'center',
                'padding': '8px'
            }).set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'center'), ('font-weight', 'bold')]}
            ]))
        else:
            print(">> No se encontraron diferencias significativas.")

    def plot_wer_boxplot(self, df_wer, title='Distribución de WER por ASR', show_stats=True):
        """
        Genera un gráfico de caja (boxplot) de la distribución de WER por ASR.
        Opcionalmente muestra la media como línea punteada y estadísticas en consola.
        """
        if 'provider' not in df_wer.columns or 'wer' not in df_wer.columns:
            raise ValueError("El DataFrame debe contener las columnas 'provider' y 'wer'.")

        providers = sorted(df_wer['provider'].unique())
        data_by_provider = [df_wer[df_wer['provider'] == p]['wer'].values for p in providers]
        palette = [self.colors.get(p, '#333333') for p in providers]

        fig, axes = plt.subplots(1, len(providers), figsize=(5 * len(providers), 6))
        if len(providers) == 1:
            axes = [axes]

        for idx, provider in enumerate(providers):
            provider_data = df_wer[df_wer['provider'] == provider]['wer']
            bp = axes[idx].boxplot(
                provider_data,
                labels=[provider.capitalize()],
                patch_artist=True,
                widths=0.6
            )
            if bp['boxes']:
                bp['boxes'][0].set_facecolor(palette[idx])
            axes[idx].set_title(f'WER - {provider.capitalize()}', fontsize=14, fontweight='bold')
            axes[idx].set_ylabel('WER (Word Error Rate)', fontsize=12)
            axes[idx].set_xlabel('ASR', fontsize=12)
            axes[idx].grid(True, alpha=0.3)

            if show_stats:
                mean_wer = provider_data.mean()
                median_wer = provider_data.median()
                axes[idx].axhline(y=mean_wer, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Media: {mean_wer:.3f}')
                axes[idx].legend(loc='upper right', fontsize=10)

        plt.tight_layout()
        plt.show()

        if show_stats:
            stats_data = []
            for provider in providers:
                provider_data = df_wer[df_wer['provider'] == provider]['wer']
                q1 = provider_data.quantile(0.25)
                q3 = provider_data.quantile(0.75)
                iqr = q3 - q1
                
                stats_data.append({
                    'ASR': provider.capitalize(),
                    'Media': provider_data.mean(),
                    'Mediana': provider_data.median(),
                    'Mínimo': provider_data.min(),
                    'Máximo': provider_data.max(),
                    'Desv. Est.': provider_data.std(),
                    'Q1': q1,
                    'Q3': q3,
                    'IQR': iqr,
                    'Muestras': len(provider_data)
                })
            
            stats_df = pd.DataFrame(stats_data)
            print("\nEstadísticas de WER por ASR:")
            return stats_df.style.format({
                'Media': '{:.4f}',
                'Mediana': '{:.4f}',
                'Mínimo': '{:.4f}',
                'Máximo': '{:.4f}',
                'Desv. Est.': '{:.4f}',
                'Q1': '{:.4f}',
                'Q3': '{:.4f}',
                'IQR': '{:.4f}'
            }).hide(axis='index').set_properties(**{
                'text-align': 'center',
                'padding': '8px'
            }).set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'center'), ('font-weight', 'bold')]}
            ])
