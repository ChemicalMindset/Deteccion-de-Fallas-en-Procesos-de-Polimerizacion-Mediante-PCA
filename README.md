**Detección de fallas en proceso de polimerización utilizando Análisis de Componentes Principales (PCA).**

Este proyecto implementa un sistema de monitoreo estadístico multivariante para un reactor con 10 (diez) sensores para variables físicas (Temperaturas, Presión, Caudal refrigerante, Agitación, pH, Viscosidad, Nivel y Flujo alimentación).

El modelo PCA se entrena con datos de operación normal y luego detecta anomalías mediante los estadísticos **T²** (distancia al subespacio PCA) y **Q** (error de reconstrucción o SPE).

Se generan 1000 muestras de operación normal y el modelo PCA selecciona el número de componentes que explican al menos el 90% de la varianza.

Una falla simulada en el sensor de temperatura T² permite validar la capacidad de detección.

---

**El script contiene:**

. Configuración de parámetros
. Función de entrenamiento PCA
. Generación de datos de prueba e inyección de falla
. Evaluación
. Gráficos

---

**Clonar repositorio:**

https://github.com/ChemicalMindset/pca-reactor-monitoring.git
cd pca-reactor-monitoring

**Crea y actica un entorno visual:**

pip install numpy matplotlib

En Windows
python -m venv venv
.\venv\Scripts\activate

En Linux/macOS:
python3 -m venv venv
source venv/bin/activate

---

**Muestra tres gráficos interactivos:**

T² vs tiempo
Q vs tiempo
Varianza explicada (barras + acumulada)

---

**Objetivos para próximas versiones:**

Incorporar una interfaz de usuario para que cada se pueda ingresar las variables correspondientes a cada proceso particular y no depender de modificar el código para nuevos controles.
