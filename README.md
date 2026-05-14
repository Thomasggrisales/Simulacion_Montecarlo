# Taller 3: Simulaciones de Monte Carlo para Optimización de Infraestructura Bioinformática

Este proyecto implementa técnicas matemáticas avanzadas de simulación de Monte Carlo para resolver problemas críticos en la gestión y análisis de infraestructura bioinformática. El objetivo principal es evaluar tiempos de ejecución, predecir fallos en pipelines de datos mediante modelado estadístico y simular tráfico de red con alta eficiencia computacional.


## Requisitos e Instalación
Para garantizar que los módulos locales se importen correctamente en el Notebook, el proyecto está configurado para instalarse en modo de desarrollo (editable).

1. Prerrequisitos:
Asegúrese de tener instalado Python 3.8 o superior. Se recomienda el uso de un entorno virtual (venv o conda).

2. Instalación de dependencias y paquete local:
Ejecute el siguiente comando desde la raíz del proyecto (donde se encuentra el archivo setup.py o pyproject.toml):

# Instala las librerías necesarias (numpy, scipy, matplotlib) 
# y vincula 'montecarlo_lib' al entorno actual
pip install -e .

### Caso 1: 
Monte Carlo Estándar (Baseline)Simulación de los tiempos de ejecución de un pipeline bioinformático modelado bajo una distribución Normal.Objetivo: Estimar la probabilidad base de que el proceso exceda un umbral de tiempo crítico.Resultado: Establece el punto de comparación teórico y demuestra la ineficiencia del método estándar para calcular probabilidades en las colas largas de la distribución (eventos raros).

### Caso 2: 
Estimación de Eventos Raros (Importance Sampling)Implementación de un estimador de varianza reducida para predecir fallos catastróficos del sistema.Técnica: Se reemplaza la distribución original por una distribución de propuesta (Exponencial desplazada) que fuerza la generación de muestras en la zona de riesgo.Resultado: Logra calcular con altísima precisión la probabilidad de un fallo crítico (del orden de 1e-02), reduciendo drásticamente la varianza del estimador y ahorrando millones de ciclos de procesamiento.

### Caso 3: 
Generación de Tráfico Sintético Bimodal (Rejection Sampling)Modelado de peticiones a un servidor que presenta dos picos de tráfico en diferentes franjas horarias (procesos nocturnos automatizados vs. tráfico de oficina).Técnica: Uso de Rejection Sampling utilizando una distribución Uniforme como envolvente $q(x)$.Optimización: Para compensar el desperdicio natural de CPU que conlleva este método matemático, se aplicó vectorización extrema utilizando NumPy. Se eliminaron los bucles iterativos en favor de operaciones matriciales simultáneas, garantizando un rendimiento casi instantáneo incluso para cientos de miles de muestras.

## Ejecución y Análisis
### Para replicar los resultados:

Abra el archivo Evaluacion_Simulacion.ipynb en Jupyter Notebook, JupyterLab o VS Code.

Seleccione la opción "Restart Kernel and Run All Cells" para asegurar una ejecución limpia.

**El Notebook generará automáticamente:**

Indicadores de eficiencia, tasas de aceptación y reducciones de varianza en notación científica.

Gráficas superpuestas que comparan las distribuciones teóricas con los histogramas de los datos generados.

Bloques de texto (Markdown) con el análisis detallado de la eficiencia computacional de cada técnica.

______________________________________________

## Autor: [Thomas Grisales]
