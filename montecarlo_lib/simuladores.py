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

    # Tiempo total 
    latencia_total = t_embeddings + t_db + t_llm

    # Cálculo de percentiles para el SLA
    p95 = np.percentile(latencia_total, 95)
    p99 = np.percentile(latencia_total, 99)

    return {
        "latencias": latencia_total,
        "percentil_95": p95,
        "percentil_99": p99
    }

import numpy as np
from scipy.stats import norm, expon

def simular_importance_sampling(
    mu_f, sigma_f, limite, num_simulaciones=100000, semilla=None
):
    if semilla is not None:
        np.random.seed(semilla)
        
    # --- Importance Sampling ---
    # Generamos muestras desde la exponencial desplazada
    muestras_is = limite + np.random.exponential(scale=1.0, size=num_simulaciones)
    
    # Calculamos densidades
    f_x = norm.pdf(muestras_is, loc=mu_f, scale=sigma_f)
    g_x = expon.pdf(muestras_is - limite, scale=1.0)
    
    # Pesos y estimaciones
    pesos = f_x / g_x
    prob_is = np.mean(pesos)
    var_is = np.var(pesos) / num_simulaciones
    
    # --- Varianza MC Estándar (Teórica) ---
    # Fórmula de varianza de una proporción (Bernoulli): p * (1 - p) / N
    var_mc_teorica = (prob_is * (1 - prob_is)) / num_simulaciones
    
    # Factor de reducción
    reduccion = var_mc_teorica / var_is if var_is > 0 else 0
    
    return {
        "prob_is": prob_is,
        "var_is": var_is,
        "var_mc_teorica": var_mc_teorica,
        "reduccion_varianza": reduccion
    }

import numpy as np

def simular_rejection_sampling(num_muestras=100000, semilla=None):
    if semilla is not None:
        np.random.seed(semilla)
        
    def p_star(x):
        pico_dia = np.exp(-0.5 * ((x - 14) / 2.5)**2) 
        pico_noche = 0.4 * np.exp(-0.5 * ((x - 2) / 1.5)**2)
        return pico_dia + pico_noche

    prob_q = 1 / 24.0
    k = 25.2
    
    # Generamos un lote gigante (10 veces más) para asegurar que 
    # sobren muestras aceptadas sin usar ciclos `for` lentos (Vectorización)
    intentos_estimados = num_muestras * 10 
    
    muestras_x = np.random.uniform(0, 24, intentos_estimados)
    muestras_y = np.random.uniform(0, k * prob_q, intentos_estimados)
    
    mascara_aceptacion = muestras_y <= p_star(muestras_x)
    muestras_aceptadas = muestras_x[mascara_aceptacion]
    
    # Recortamos exactamente a la cantidad solicitada
    muestras_finales = muestras_aceptadas[:num_muestras]
    
    # Calculamos la tasa de aceptación real (Total aceptadas / Total intentadas)
    tasa_aceptacion = len(muestras_aceptadas) / intentos_estimados
    
    return {
        "muestras": muestras_finales,
        "tasa_aceptacion": tasa_aceptacion,
        "k": k,
        "funcion_p_star": p_star
    }