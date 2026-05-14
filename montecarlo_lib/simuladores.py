import numpy as np

def simular_latencia_rag(
    mu_emb, sigma_emb,
    low_db, high_db,
    mean_llm, sigma_llm,
    num_simulaciones=100000,
    semilla=None
):
    """
    Simula el tiempo de respuesta de un sistema RAG usando el Método de Montecarlo Estándar.

    Parameters:
    -----------
    mu_emb : float
        Media de la distribución Normal para la generación de embeddings.
    sigma_emb : float
        Desviación estándar de la distribución Normal para embeddings.
    low_db : float
        Límite inferior de la distribución Uniforme para la búsqueda en BD.
    high_db : float
        Límite superior de la distribución Uniforme para la búsqueda en BD.
    mean_llm : float
        Media de la distribución Normal subyacente para la inferencia del LLM (Lognormal).
    sigma_llm : float
        Desviación estándar de la distribución Normal subyacente para el LLM.
    num_simulaciones : int, opcional
        Número de consultas a simular (por defecto 100000).
    semilla : int, opcional
        Semilla de reproducibilidad para los números aleatorios.

    Returns:
    --------
    dict
        Diccionario con las latencias totales simuladas y los percentiles 95 y 99.
    """
    if semilla is not None:
        np.random.seed(semilla)

    # 1. Generación de embeddings (Distribución Normal)
    # Usamos np.maximum para evitar tiempos negativos irreales
    t_embeddings = np.maximum(np.random.normal(mu_emb, sigma_emb, num_simulaciones), 0)

    # 2. Búsqueda en la base de datos vectorial (Distribución Uniforme)
    t_db = np.random.uniform(low_db, high_db, num_simulaciones)

    # 3. Inferencia del LLM (Distribución Lognormal)
    t_llm = np.random.lognormal(mean_llm, sigma_llm, num_simulaciones)

    # Tiempo total (Suma vectorizada de arreglos para máxima eficiencia)
    latencia_total = t_embeddings + t_db + t_llm

    # Cálculo de percentiles para el SLA
    p95 = np.percentile(latencia_total, 95)
    p99 = np.percentile(latencia_total, 99)

    return {
        "latencias": latencia_total,
        "percentil_95": p95,
        "percentil_99": p99
    }