"""
Интерактивные графики для визуализации Shapley values с использованием Plotly
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly не установлен. Интерактивные графики недоступны.")


def plot_shap_importance_interactive(
    shap_values: np.ndarray,
    feature_names: Optional[List[str]] = None,
    title: str = "Shapley Values - Важность признаков",
    output_path: Optional[str] = None,
    show: bool = True
) -> go.Figure:
    """
    Создает интерактивный график важности признаков на основе Shapley values
    
    Args:
        shap_values: Shapley values для каждого признака
        feature_names: Названия признаков (если None, используются индексы)
        title: Заголовок графика
        output_path: Путь для сохранения HTML файла
        show: Показывать ли график в браузере
        
    Returns:
        go.Figure: Интерактивный график Plotly
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly не установлен. Установите: pip install plotly")
    
    if feature_names is None:
        feature_names = [f"Признак {i+1}" for i in range(len(shap_values))]
    
    # Сортируем по важности
    sorted_indices = np.argsort(np.abs(shap_values))[::-1]
    sorted_values = shap_values[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]
    
    # Цвета: положительные - синие, отрицательные - красные
    colors = ['#5B8DEF' if v >= 0 else '#F17666' for v in sorted_values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=sorted_names,
        y=sorted_values,
        marker_color=colors,
        text=[f"{v:.4f}" for v in sorted_values],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Shapley Value: %{y:.6f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Признаки",
        yaxis_title="Shapley Value",
        xaxis={'categoryorder': 'total descending'},
        hovermode='closest',
        template='plotly_white',
        height=500,
        showlegend=False
    )
    
    if output_path:
        fig.write_html(output_path)
        print(f"✅ График сохранен: {output_path}")
    
    if show:
        fig.show()
    
    return fig


def plot_shap_waterfall_interactive(
    shap_values: np.ndarray,
    feature_names: Optional[List[str]] = None,
    baseline_value: float = 0.0,
    prediction_value: Optional[float] = None,
    title: str = "Shapley Values - Waterfall Plot",
    output_path: Optional[str] = None,
    show: bool = True
) -> go.Figure:
    """
    Создает интерактивный waterfall график Shapley values
    
    Args:
        shap_values: Shapley values для каждого признака
        feature_names: Названия признаков
        baseline_value: Базовое значение (обычно среднее предсказание)
        prediction_value: Финальное предсказание (если None, вычисляется как baseline + sum(shap))
        title: Заголовок графика
        output_path: Путь для сохранения HTML файла
        show: Показывать ли график в браузере
        
    Returns:
        go.Figure: Интерактивный waterfall график
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly не установлен. Установите: pip install plotly")
    
    if feature_names is None:
        feature_names = [f"Признак {i+1}" for i in range(len(shap_values))]
    
    # Сортируем по абсолютному значению
    sorted_indices = np.argsort(np.abs(shap_values))[::-1]
    sorted_values = shap_values[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]
    
    # Вычисляем кумулятивные значения
    cumulative = baseline_value + np.cumsum(sorted_values)
    
    if prediction_value is None:
        prediction_value = cumulative[-1]
    
    # Создаем данные для waterfall
    x_data = ['Базовое значение'] + sorted_names + ['Предсказание']
    y_data = [baseline_value] + sorted_values.tolist() + [0]
    
    # Цвета
    colors = ['#E8E8E8']  # Базовое значение - серое
    colors.extend(['#5B8DEF' if v >= 0 else '#F17666' for v in sorted_values])
    colors.append('#2ECC71')  # Предсказание - зеленое
    
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute"] + ["relative"] * len(sorted_values) + ["total"],
        x=x_data,
        y=y_data,
        textposition="outside",
        text=[f"{v:.4f}" for v in y_data],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#5B8DEF"}},
        decreasing={"marker": {"color": "#F17666"}},
        totals={"marker": {"color": "#2ECC71"}}
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Признаки",
        yaxis_title="Вклад в предсказание",
        template='plotly_white',
        height=600,
        showlegend=False
    )
    
    if output_path:
        fig.write_html(output_path)
        print(f"✅ Waterfall график сохранен: {output_path}")
    
    if show:
        fig.show()
    
    return fig


def plot_shap_metrics_interactive(
    metrics: Dict,
    output_path: Optional[str] = None,
    show: bool = True
) -> go.Figure:
    """
    Создает интерактивный график метрик качества Shapley values
    
    Args:
        metrics: Словарь с метриками (stability, faithfulness, efficiency, symmetry)
        output_path: Путь для сохранения HTML файла
        show: Показывать ли график в браузере
        
    Returns:
        go.Figure: Интерактивный график метрик
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly не установлен. Установите: pip install plotly")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Стабильность', 'Faithfulness', 'Эффективность', 'Симметричность'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "indicator"}, {"type": "bar"}]]
    )
    
    # Стабильность
    if 'stability' in metrics:
        stability = metrics['stability']
        fig.add_trace(
            go.Bar(
                x=['Mean', 'Max', 'Min'],
                y=[stability.get('mean_stability', 0),
                   stability.get('max_stability', 0),
                   stability.get('min_stability', 0)],
                name='Стабильность',
                marker_color='#5B8DEF'
            ),
            row=1, col=1
        )
    
    # Faithfulness
    if 'faithfulness' in metrics:
        faithfulness = metrics['faithfulness']
        fig.add_trace(
            go.Bar(
                x=['Deletion', 'Insertion', 'Overall'],
                y=[faithfulness.get('mean_deletion', 0),
                   faithfulness.get('mean_insertion', 0),
                   faithfulness.get('faithfulness_score', 0)],
                name='Faithfulness',
                marker_color='#F17666'
            ),
            row=1, col=2
        )
    
    # Эффективность (индикатор)
    if 'efficiency' in metrics:
        efficiency = metrics['efficiency']
        is_efficient = efficiency.get('is_efficient', False)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=1 if is_efficient else 0,
                title={'text': 'Эффективность'},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': '#2ECC71' if is_efficient else '#F17666'},
                    'steps': [
                        {'range': [0, 0.5], 'color': '#F17666'},
                        {'range': [0.5, 1], 'color': '#2ECC71'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.5
                    }
                }
            ),
            row=2, col=1
        )
    
    # Симметричность
    if 'symmetry' in metrics:
        symmetry = metrics['symmetry']
        fig.add_trace(
            go.Bar(
                x=['Mean Error', 'Max Error'],
                y=[symmetry.get('mean_symmetry_error', 0),
                   symmetry.get('max_symmetry_error', 0)],
                name='Симметричность',
                marker_color='#9B59B6'
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title_text="Метрики качества Shapley Values",
        template='plotly_white',
        height=800,
        showlegend=False
    )
    
    if output_path:
        fig.write_html(output_path)
        print(f"✅ График метрик сохранен: {output_path}")
    
    if show:
        fig.show()
    
    return fig


def plot_shap_comparison_interactive(
    shap_values_list: List[np.ndarray],
    labels: List[str],
    feature_names: Optional[List[str]] = None,
    title: str = "Сравнение Shapley Values",
    output_path: Optional[str] = None,
    show: bool = True
) -> go.Figure:
    """
    Создает интерактивный график сравнения Shapley values для разных моделей
    
    Args:
        shap_values_list: Список массивов Shapley values
        labels: Метки для каждого набора значений
        feature_names: Названия признаков
        title: Заголовок графика
        output_path: Путь для сохранения HTML файла
        show: Показывать ли график в браузере
        
    Returns:
        go.Figure: Интерактивный график сравнения
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly не установлен. Установите: pip install plotly")
    
    if feature_names is None:
        n_features = len(shap_values_list[0])
        feature_names = [f"Признак {i+1}" for i in range(n_features)]
    
    fig = go.Figure()
    
    colors = ['#5B8DEF', '#F17666', '#2ECC71', '#9B59B6', '#F39C12']
    
    for i, (shap_values, label) in enumerate(zip(shap_values_list, labels)):
        fig.add_trace(go.Bar(
            x=feature_names,
            y=shap_values,
            name=label,
            marker_color=colors[i % len(colors)],
            hovertemplate=f'<b>{label}</b><br>%{{x}}: %{{y:.6f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Признаки",
        yaxis_title="Shapley Value",
        barmode='group',
        hovermode='closest',
        template='plotly_white',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    if output_path:
        fig.write_html(output_path)
        print(f"✅ График сравнения сохранен: {output_path}")
    
    if show:
        fig.show()
    
    return fig

