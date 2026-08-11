**Detección de fallas en proceso de polimerización utilizando Análisis de Componentes Principales (PCA).**

Este proyecto implementa un sistema de monitoreo estadístico multivariante para un reactor con 10 (diez) sensores para variables físicas (Temperaturas, Presión, Caudal refrigerante, Agitación, pH, Viscosidad, Nivel y Flujo alimentación).

El modelo PCA se entrena con datos de operación normal y luego detecta anomalías mediante los estadísticos **T²** (distancia al subespacio PCA) y **Q** (error de reconstrucción o SPE).

Una falla simulada en el sensor de temperatura T² permite validar la capacidad de detección.

---
**El script contiene:**

. Configuración de parámetros
. Función de entrenamiento PCA
. Generación de datos de prueba e inyección de falla
. Evaluación
. Gráficos

---

Clonar repositorio
