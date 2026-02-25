"""
Análisis y visualización de resultados CER (Character Error Rate).
Responsabilidad: agregaciones por provider/snr/category y presentación en tablas.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import itertools
from IPython.display import display, HTML


class CERAnalyzer:
    """
    Responsable de agregar los datos de CER y calcular métricas por grupo
    (provider, snr, category; solo category; snr + category).
    """

    def __init__(self, cer_df: pd.DataFrame):
        """
        Args:
            cer_df: DataFrame con columnas edit_distance, ref_length, provider, snr, category.
        """
        self.cer_df = cer_df.copy()

    def perform_statistical_tests(self, category: str = None) -> tuple:
        """
        Realiza pruebas estadísticas (Friedman y Wilcoxon post-hoc) para comparar ASR.
        Si se especifica category, filtra por esa categoría.
        
        Returns:
            tuple: (friedman_result, wilcoxon_results_df)
                   friedman_result: (statistic, p_value)
                   wilcoxon_results_df: DataFrame con resultados post-hoc o None.
        """
        df_target = self.cer_df.copy()
        if category:
            df_target = df_target[df_target['category'] == category]
            
        # Crear ID único para pivotar: audio + noise + snr + category + instance_id
        # Asumiendo que esta combinación es única por proveedor
        # Generar un identificador de instancia para diferenciar múltiples keywords de la misma categoría en el mismo audio.
        df_target['instance_id'] = df_target.groupby(['provider', 'audio', 'noise', 'snr', 'category']).cumcount()

        df_target['unique_id'] = (
            df_target['audio'].astype(str) + '_' + 
            df_target['noise'].astype(str) + '_' + 
            df_target['snr'].astype(str) + '_' + 
            df_target['category'].astype(str) + '_' +
            df_target['instance_id'].astype(str)
        )
        
        # Calcular CER por muestra
        def safe_cer(row):
            return row['edit_distance'] / row['ref_length'] if row['ref_length'] > 0 else 0.0
            
        df_target['sample_cer'] = df_target.apply(safe_cer, axis=1)
        
        # Pivotar: filas=muestras, columnas=proveedores
        pivot_cer = df_target.pivot(index='unique_id', columns='provider', values='sample_cer').dropna()
        
        if pivot_cer.empty:
            print(f"No hay datos suficientes para análisis estadístico en categoría: {category}")
            return (None, None), None
            
        # Test de Friedman
        stat, p_value = stats.friedmanchisquare(*[pivot_cer[col] for col in pivot_cer.columns])
        friedman_result = (stat, p_value)
        
        wilcoxon_df = None
        if p_value < 0.05:
            comparisons = []
            p_values = []
            effect_sizes = []
            pairs = list(itertools.combinations(pivot_cer.columns, 2))
            n_comparisons = len(pairs)
            
            for p1, p2 in pairs:
                # Wilcoxon test
                try:
                    stat_w, p_w = stats.wilcoxon(pivot_cer[p1], pivot_cer[p2])
                except ValueError:
                    # Todos las diferencias son cero
                    stat_w, p_w = 0, 1.0
                    
                comparisons.append(f"{p1} vs {p2}")
                p_values.append(p_w)
                
                # Tamaño del efecto (r = Z / sqrt(N))
                n = len(pivot_cer)
                mu = n * (n + 1) / 4
                se = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
                
                if se > 0:
                    z = (stat_w - mu) / se
                    r = abs(z) / np.sqrt(n)
                else:
                    r = 0.0
                effect_sizes.append(r)
            
            # Ajuste Holm-Bonferroni (igual que en WER)
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
                'p-value original': p_values,
                'p-value adj (Holm-Bonferroni)': p_adjusted,
                'Significativo': reject,
                'Tamaño del Efecto (r)': effect_sizes
            })
            
        return friedman_result, wilcoxon_df

    def check_normality(self, provider: str = None, category: str = None) -> None:
        """
        Realiza una prueba de normalidad (Shapiro-Wilk) y genera gráficos Q-Q e histograma
        para evaluar la distribución de los errores (CER por muestra).
        
        Args:
            provider: Filtrar por proveedor (opcional).
            category: Filtrar por categoría (opcional).
        """
        df_subset = self.cer_df.copy()
        if provider:
            df_subset = df_subset[df_subset['provider'] == provider]
        if category:
            df_subset = df_subset[df_subset['category'] == category]
            
        if df_subset.empty:
            print("No hay datos para los filtros seleccionados.")
            return

        # Calcular CER por muestra
        # Evitar división por cero
        def safe_cer(row):
            return row['edit_distance'] / row['ref_length'] if row['ref_length'] > 0 else 0.0
            
        data = df_subset.apply(safe_cer, axis=1).dropna()
        
        if len(data) < 3:
            print("Insuficientes datos para prueba de normalidad (N < 3).")
            return
            
        # Shapiro-Wilk Test
        stat, p_value = stats.shapiro(data)
        
        print(f"--- Prueba de Normalidad (Shapiro-Wilk) ---")
        if provider: print(f"Provider: {provider}")
        if category: print(f"Category: {category}")
        print(f"N: {len(data)}")
        print(f"Estadístico W: {stat:.4f}")
        print(f"p-value: {p_value:.4e}")
        
        if p_value < 0.05:
            print(">> Se rechaza la hipótesis nula: Los datos NO siguen una distribución normal (p < 0.05).")
        else:
            print(">> No se rechaza la hipótesis nula: Los datos podrían seguir una distribución normal (p >= 0.05).")
            
        # Visualización
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histograma
        axes[0].hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_title('Histograma de CER')
        axes[0].set_xlabel('CER')
        axes[0].set_ylabel('Frecuencia')
        
        # Q-Q Plot
        stats.probplot(data, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot')
        
        plt.tight_layout()
        plt.show()

    def bootstrap_cer_ci(self, df_group: pd.DataFrame, n_bootstraps: int = 1000, confidence_level: float = 0.95) -> tuple:
        """
        Calcula el intervalo de confianza del CER usando bootstrap.
        
        Args:
            df_group: DataFrame del grupo (debe tener edit_distance y ref_length).
            n_bootstraps: Número de muestras bootstrap.
            confidence_level: Nivel de confianza (0.95 por defecto).
            
        Returns:
            (lower_bound, upper_bound) del CER.
        """
        if df_group.empty:
            return 0.0, 0.0

        distances = df_group['edit_distance'].values
        lengths = df_group['ref_length'].values
        n_samples = len(distances)
        
        cer_samples = []
        rng = np.random.default_rng(42)
        
        for _ in range(n_bootstraps):
            indices = rng.integers(0, n_samples, n_samples)
            sample_dist = distances[indices].sum()
            sample_len = lengths[indices].sum()
            cer = sample_dist / sample_len if sample_len > 0 else 0.0
            cer_samples.append(cer)
            
        alpha = (1 - confidence_level) / 2
        lower = np.percentile(cer_samples, alpha * 100)
        upper = np.percentile(cer_samples, (1 - alpha) * 100)
        
        return lower, upper

    def get_cer_by_provider_and_category_with_ci(self) -> pd.DataFrame:
        """
        Calcula CER promedio y sus intervalos de confianza por proveedor y categoría.
        
        Returns:
            DataFrame con columnas: provider, category, cer, ci_lower, ci_upper, cer_pct, ci_lower_pct, ci_upper_pct, count.
        """
        results = []
        
        # Agrupar por provider y category
        groups = self.cer_df.groupby(['provider', 'category'])
        
        for (provider, category), group in groups:
            # Cálculo puntual
            total_dist = group['edit_distance'].sum()
            total_len = group['ref_length'].sum()
            cer = total_dist / total_len if total_len > 0 else 0.0
            
            # Cálculo de IC
            lower, upper = self.bootstrap_cer_ci(group)
            
            results.append({
                'provider': provider,
                'category': category,
                'cer': cer,
                'ci_lower': lower,
                'ci_upper': upper,
                'cer_pct': cer * 100,
                'ci_lower_pct': lower * 100,
                'ci_upper_pct': upper * 100,
                'count': len(group)
            })
            
        return pd.DataFrame(results)

    def get_global_cer_by_provider_with_ci(self) -> pd.DataFrame:
        """
        Calcula CER promedio global y sus intervalos de confianza por proveedor (agrupando todas las categorías).
        
        Returns:
            DataFrame con columnas: provider, cer, ci_lower, ci_upper, cer_pct, ci_lower_pct, ci_upper_pct, count.
        """
        results = []
        
        # Agrupar solo por provider
        groups = self.cer_df.groupby(['provider'])
        
        for provider, group in groups:
            # Si provider es una tupla (comportamiento de groupby con lista), extraer el valor
            if isinstance(provider, tuple):
                provider = provider[0]
                
            # Cálculo puntual
            total_dist = group['edit_distance'].sum()
            total_len = group['ref_length'].sum()
            cer = total_dist / total_len if total_len > 0 else 0.0
            
            # Cálculo de IC
            lower, upper = self.bootstrap_cer_ci(group)
            
            results.append({
                'provider': provider,
                'cer': cer,
                'ci_lower': lower,
                'ci_upper': upper,
                'cer_pct': cer * 100,
                'ci_lower_pct': lower * 100,
                'ci_upper_pct': upper * 100,
                'count': len(group)
            })
            
        return pd.DataFrame(results)

    def get_grouped_by_provider_snr_category(self) -> pd.DataFrame:
        """
        Agrupa por ASR (provider), SNR y categoría; suma edit_distance y ref_length;
        calcula CER y CER en porcentaje.

        Returns:
            DataFrame con columnas provider, snr, category, edit_distance, ref_length, cer, cer_pct.
        """
        grouped = self.cer_df.groupby(["provider", "snr", "category"])[
            ["edit_distance", "ref_length"]
        ].sum().reset_index()
        grouped["cer"] = grouped["edit_distance"] / grouped["ref_length"]
        grouped["cer_pct"] = grouped["cer"] * 100
        return grouped

    def get_global_by_category(self) -> pd.DataFrame:
        """
        CER global por categoría (sin desglose por provider ni SNR).

        Returns:
            DataFrame con columnas category, edit_distance, ref_length, cer, cer_pct.
        """
        g = self.cer_df.groupby("category")[["edit_distance", "ref_length"]].sum().reset_index()
        g["cer"] = g["edit_distance"] / g["ref_length"]
        g["cer_pct"] = g["cer"] * 100
        return g

    def get_cer_by_snr_and_category(self) -> pd.DataFrame:
        """
        CER por nivel de ruido (SNR) y categoría, en formato pivot para visualización
        (columnas: category, cer clean, cer 10db, cer 5db, cer 0db).

        Returns:
            DataFrame pivot con categoría como índice y columnas de CER por SNR.
        """
        grouped = self.cer_df.groupby(["snr", "category"])[["edit_distance", "ref_length"]].sum().reset_index()
        grouped["cer_pct"] = grouped["edit_distance"] / grouped["ref_length"] * 100

        cer_by_snr = grouped.pivot_table(
            index="category",
            columns="snr",
            values="cer_pct",
            aggfunc="first",
        ).reset_index()

        col_map = {"clean": "cer clean", "10dB": "cer 10db", "5dB": "cer 5db", "0dB": "cer 0db"}
        cer_by_snr = cer_by_snr.rename(columns=col_map)
        requested = ["category", "cer clean", "cer 10db", "cer 5db", "cer 0db"]
        return cer_by_snr[[c for c in requested if c in cer_by_snr.columns]]


class CERVisualizer:
    """
    Responsable de mostrar tablas y resultados CER en el notebook (HTML, display).
    """

    @staticmethod
    def display_global_by_category(df: pd.DataFrame, title: str = "Global CER by Category") -> None:
        """Muestra la tabla de CER global por categoría como HTML."""
        display(HTML(f"<h3>{title}</h3>"))
        display(HTML(df.to_html(index=False)))

    @staticmethod
    def display_cer_by_snr(
        df: pd.DataFrame,
        title: str = "Variación del CER por nivel de ruido",
    ) -> None:
        """Muestra la tabla de CER por SNR y categoría como HTML."""
        display(HTML(f"<h3>{title}</h3>"))
        display(HTML(df.to_html(index=False)))

    @staticmethod
    def plot_cer_by_category(df_stats: pd.DataFrame, title: str = "CER por Categoría y Proveedor") -> None:
        """
        Genera un gráfico de barras agrupadas con barras de error para el CER por categoría.
        
        Args:
            df_stats: DataFrame con columnas provider, category, cer_pct, ci_lower_pct, ci_upper_pct.
        """
        if df_stats.empty:
            print("No hay datos para graficar.")
            return

        categories = sorted(df_stats['category'].unique())
        providers = sorted(df_stats['provider'].unique())
        
        n_categories = len(categories)
        n_providers = len(providers)
        
        width = 0.8 / n_providers
        x = np.arange(n_categories)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = {
            'google': '#4285F4',
            'azure': '#0078D4',
            'amazon': '#FF9900',
            'whisper': '#34A853'
        }
        
        for i, provider in enumerate(providers):
            p_data = df_stats[df_stats['provider'] == provider]
            # Asegurar orden de categorías
            p_data = p_data.set_index('category').reindex(categories).reset_index()
            
            means = p_data['cer_pct'].fillna(0)
            
            # Calcular errores asimétricos
            yerr_lower = means - p_data['ci_lower_pct'].fillna(0)
            yerr_upper = p_data['ci_upper_pct'].fillna(0) - means
            # Evitar errores negativos por redondeo
            yerr_lower = np.maximum(yerr_lower, 0)
            yerr_upper = np.maximum(yerr_upper, 0)
            
            offset = (i - n_providers/2 + 0.5) * width
            
            provider_label = str(provider).capitalize()
            ax.bar(x + offset, means, width, label=provider_label,
                   yerr=[yerr_lower, yerr_upper], capsize=5, 
                   color=colors.get(provider, None), alpha=0.9, edgecolor='white')

        ax.set_ylabel('CER Promedio (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Categoría de Entidad', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=11)
        ax.legend(title='ASR', fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_global_cer(df_stats: pd.DataFrame, title: str = "CER Global por Proveedor") -> None:
        """
        Genera un gráfico de barras simple con barras de error para el CER global por proveedor.
        
        Args:
            df_stats: DataFrame con columnas provider, cer_pct, ci_lower_pct, ci_upper_pct.
        """
        if df_stats.empty:
            print("No hay datos para graficar.")
            return

        providers = sorted(df_stats['provider'].unique())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = {
            'google': '#4285F4',
            'azure': '#0078D4',
            'amazon': '#FF9900',
            'whisper': '#34A853'
        }
        
        # Prepare data
        means = []
        yerr_lower = []
        yerr_upper = []
        bar_colors = []
        
        for provider in providers:
            row = df_stats[df_stats['provider'] == provider].iloc[0]
            mean = row['cer_pct']
            lower = row['ci_lower_pct']
            upper = row['ci_upper_pct']
            
            means.append(mean)
            yerr_lower.append(max(0, mean - lower))
            yerr_upper.append(max(0, upper - mean))
            bar_colors.append(colors.get(provider, 'gray'))
            
        x = np.arange(len(providers))
        
        ax.bar(x, means, yerr=[yerr_lower, yerr_upper], capsize=10, 
               color=bar_colors, alpha=0.9, edgecolor='white', width=0.6)
        
        ax.set_ylabel('CER Promedio (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('ASR', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(p).capitalize() for p in providers], fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Add value labels
        for i, v in enumerate(means):
            ax.text(i, v + yerr_upper[i] + 0.5, f"{v:.2f}%", ha='center', fontweight='bold')
            
        plt.tight_layout()
        plt.show()

    @staticmethod
    def display_global_cer_table(summary_df: pd.DataFrame):
        """
        Muestra la tabla de CER global formateada (estilo WER: ASR, CER Global, IC 95% Inf, IC 95% Sup).
        """
        display_df = summary_df.rename(columns={
            'provider': 'ASR',
            'cer_pct': 'CER Global',
            'ci_lower_pct': 'IC 95% Inf',
            'ci_upper_pct': 'IC 95% Sup'
        })
        print("Tabla Comparativa: Rendimiento Global con Intervalos de Confianza")
        return display_df[['ASR', 'CER Global', 'IC 95% Inf', 'IC 95% Sup']].style.format({
            'CER Global': '{:.2f}%',
            'IC 95% Inf': '{:.2f}%',
            'IC 95% Sup': '{:.2f}%'
        }).hide(axis='index')

    @staticmethod
    def display_global_statistical_results(friedman_result, wilcoxon_df, title: str = "Análisis Estadístico de Significancia (CER Global)") -> None:
        """
        Muestra los resultados de Friedman y post-hoc (Wilcoxon + Holm-Bonferroni) para CER global.
        Formato similar a 2_wer.ipynb.
        """
        stat, p_value = friedman_result
        if stat is None:
            print(f"\n{title}")
            print("No se pudieron realizar las pruebas estadísticas (datos insuficientes o estructura incorrecta).")
            return

        print(f"\n{title}")
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

    @staticmethod
    def display_cer_statistics(df_stats: pd.DataFrame, title: str = "Estadísticas Descriptivas de CER") -> None:
        """
        Muestra una tabla con CER medio e intervalos de confianza.
        Formato: "Media% [IC 95%: Lower% - Upper%] (N=...)"
        """
        if df_stats.empty:
            print("No hay datos para mostrar.")
            return

        # Formatear celda
        def format_cell(row):
            count_str = f" (N={row['count']})" if 'count' in row else ""
            return f"{row['cer_pct']:.2f}% [{row['ci_lower_pct']:.2f}% - {row['ci_upper_pct']:.2f}%]{count_str}"

        df_formatted = df_stats.copy()
        df_formatted['formatted'] = df_formatted.apply(format_cell, axis=1)
        
        # Pivotar
        pivot_table = df_formatted.pivot(index='category', columns='provider', values='formatted')
        
        display(HTML(f"<h3>{title}</h3>"))
        display(HTML(pivot_table.to_html()))

    @staticmethod
    def display_statistical_results(results: dict, title: str = "Resultados de Significancia Estadística") -> None:
        """
        Muestra los resultados de las pruebas estadísticas (Friedman y Wilcoxon) en formato tabla.
        
        Args:
            results: Diccionario {category: (friedman_result, wilcoxon_df)}
        """
        if not results:
            print("No hay resultados estadísticos para mostrar.")
            return

        display(HTML(f"<h3>{title}</h3>"))
        
        for category, (friedman, wilcoxon) in results.items():
            stat, p_val = friedman
            if stat is None: continue
            
            display(HTML(f"<h4>Categoría: {category}</h4>"))
            display(HTML(f"<p><b>Test de Friedman:</b> Chi2={stat:.4f}, p-value={p_val:.4e}</p>"))
            
            if wilcoxon is not None:
                display(HTML(wilcoxon.to_html(index=False)))
            else:
                display(HTML("<p>No se encontraron diferencias significativas (p >= 0.05).</p>"))

    @staticmethod
    def plot_cer_boxplot(cer_df: pd.DataFrame, category: str = None, title: str = "Distribución de CER por ASR") -> None:
        """
        Genera un gráfico de cajas (boxplot) para la distribución de CER.
        
        Args:
            cer_df: DataFrame original con datos de CER.
            category: Categoría específica a graficar (opcional).
        """
        df_plot = cer_df.copy()
        if category:
            df_plot = df_plot[df_plot['category'] == category]
            title += f" - {category}"
            
        # Calcular CER por muestra
        def safe_cer(row):
            return row['edit_distance'] / row['ref_length'] if row['ref_length'] > 0 else 0.0
            
        df_plot['sample_cer'] = df_plot.apply(safe_cer, axis=1)
        
        providers = sorted(df_plot['provider'].unique())
        data_by_provider = [df_plot[df_plot['provider'] == p]['sample_cer'].values for p in providers]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bp = ax.boxplot(data_by_provider, labels=[str(p).capitalize() for p in providers], patch_artist=True)
        
        colors = ['#4285F4', '#0078D4', '#FF9900', '#34A853'] # Google, Azure, Amazon, Whisper
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            
        ax.set_ylabel('CER (Character Error Rate)', fontsize=12, fontweight='bold')
        ax.set_xlabel('ASR', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
