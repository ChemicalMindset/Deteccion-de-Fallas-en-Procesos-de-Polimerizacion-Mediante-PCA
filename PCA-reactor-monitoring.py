# Reactor de polimerización con 10 sensores

# Librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from numpy.typing import NDArray
from typing import TypedDict

# ============================================================
# 1. DEFINICIÓN DE PARÁMETROS FÍSICOS
# ============================================================

class ReactorParams(TypedDict):
    means: NDArray[np.float64]
    std_devs: NDArray[np.float64]
    correlations: NDArray[np.float64]
    n_features: int
    n_samples: int

def configurar_reactor() -> ReactorParams:

    # Parámetros de operación normal
    # Medidas: T1, T2, T3, Presión, Caudal refrigerante, Agitación, pH, Viscosidad, Nivel, Flujo alimentación
    
    params: ReactorParams = {
    "means": np.array([80.0, 85.0, 82.0, 5.2, 120.0, 3.1, 7.0, 1.2, 0.8, 0.5]),
    # Desviación estándar de cada sensor
    "std_devs": np.array([2.0, 2.5, 2.0, 0.5, 10.0, 0.4, 0.3, 0.2, 0.1, 0.1]),
    # Matriz de correlaciones
    "correlations": np.array([
        [1.0, 0.9, 0.8, 0.6, 0.2, 0.1, 0.3, -0.7, 0.4, 0.5],
        [0.9, 1.0, 0.9, 0.7, 0.3, 0.2, 0.4, -0.8, 0.5, 0.6],
        [0.8, 0.9, 1.0, 0.8, 0.4, 0.3, 0.5, -0.9, 0.6, 0.7],
        [0.6, 0.7, 0.8, 1.0, 0.5, 0.4, 0.6, -0.6, 0.7, 0.8],
        [0.2, 0.3, 0.4, 0.5, 1.0, 0.8, 0.2, -0.3, 0.4, 0.5],
        [0.1, 0.2, 0.3, 0.4, 0.8, 1.0, 0.1, -0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5, 0.6, 0.2, 0.1, 1.0, -0.4, 0.5, 0.6],
        [-0.7, -0.8, -0.9, -0.6, -0.3, -0.2, -0.4, 1.0, -0.5, -0.6],
        [0.4, 0.5, 0.6, 0.7, 0.4, 0.3, 0.5, -0.5, 1.0, 0.9],
        [0.5, 0.6, 0.7, 0.8, 0.5, 0.4, 0.6, -0.6, 0.9, 1.0]
    ]),
    # Número de sensores y muestras de operación.
    "n_features": 10,
    "n_samples": 1000
    }
    return params

def construir_covarianza(
    std_devs: NDArray[np.float64],
    correlations: NDArray[np.float64],
) -> NDArray[np.float64]:
    cov_matrix = np.outer(std_devs, std_devs) * correlations
    return cov_matrix

# ============================================================
# 2. MODELO PCA
# ============================================================

class PCA_Model(TypedDict):
    mean_train: NDArray[np.float64]
    std_train: NDArray[np.float64]
    V: NDArray[np.float64]
    k: int
    eigenvalues: NDArray[np.float64]
    lambda_inv: NDArray[np.float64]
    T2_limit: float
    Q_limit: float
    var_expl: NDArray[np.float64]
    var_ratio: NDArray[np.float64]
    var_acum: NDArray[np.float64]
    scores_train: NDArray[np.float64]
    T2_train: NDArray[np.float64]
    Q_train: NDArray[np.float64]
    var_threshold: NDArray[np.float64]

def entrenar_pca(
    X_train: NDArray[np.float64],
    var_threshold: float = 0.95
) -> PCA_Model:


    # --------------------------------------------------------
    # 2.1 Estandarización
    # --------------------------------------------------------

    mean_train = np.mean(X_train, axis = 0)
    std_train = np.std(X_train, axis=0, ddof=1)
    X_scaled = (X_train - mean_train) / std_train

    # --------------------------------------------------------
    # 2.2 Descomposición SVD
    # --------------------------------------------------------

    _, S_raw, Vt_raw = np.linalg.svd(X_scaled, full_matrices=False)
    S: NDArray[np.float64] = np.asarray(S_raw, dtype=np.float64)
    Vt: NDArray[np.float64] = np.asarray(Vt_raw, dtype=np.float64)
    V = Vt.T # V contiene los vectores principales

    # --------------------------------------------------------
    # 2.3 Varianza explicada
    # --------------------------------------------------------

    n_samples = X_train.shape[0]
    var_expl: NDArray[np.float64] = (S ** 2) / float(n_samples - 1)
    var_ratio: NDArray[np.float64] = var_expl / np.sum(var_expl)
    var_acum: NDArray[np.float64] = np.cumsum(var_ratio)

    # --------------------------------------------------------
    # 2.4 Selección automática de componentes
    # --------------------------------------------------------

    k = int(np.argmax(var_acum >= var_threshold) + 1)

    # --------------------------------------------------------
    # 2.5 Scores
    # --------------------------------------------------------

    scores_train = (
        X_scaled
        @ V[:, :k]
    )

    # --------------------------------------------------------
    # 2.6 Estadístico T²
    # --------------------------------------------------------

    eigenvalues = (S[:k] ** 2 / float(n_samples - 1))
    lambda_inv = 1.0/ eigenvalues
    T2_train = np.sum((scores_train ** 2) * lambda_inv, axis=1)

    # --------------------------------------------------------
    # 2.7 Estadístico Q / SPE
    # --------------------------------------------------------

    X_reconstructed = (
        scores_train
        @ V[:, :k].T
    )
    residuos_train = (X_scaled - X_reconstructed)
    Q_train = np.sum(residuos_train ** 2, axis=1)

    # --------------------------------------------------------
    # 2.8 Límites de control
    # --------------------------------------------------------

    T2_limit = float(np.percentile(T2_train, 95))
    Q_limit = float(np.percentile(Q_train, 95))

    # --------------------------------------------------------
    # 2.9 Guardar modelo
    # --------------------------------------------------------

    pca_modelo: PCA_Model = {
        "mean_train": mean_train,
        "std_train": std_train,
        "V": V,
        #"k": int(np.argmax(var_acum >= var_threshold) + 1),
        "k": k,
        "eigenvalues": eigenvalues,
        "lambda_inv": lambda_inv,
        "T2_limit": T2_limit,
        "Q_limit": Q_limit,
        "var_expl": var_expl,
        "var_ratio": var_ratio,
        "var_acum": var_acum,
        "scores_train": scores_train,
        "T2_train": T2_train,
        "Q_train": Q_train,
        "var_threshold": np.array([var_threshold], dtype=np.float64),
    }
    return pca_modelo


# ============================================================
# 3. EJECUCIÓN PRINCIPAL
# ============================================================

def main():

    # --------------------------------------------------------
    # 3.1 Generación de datos
    # --------------------------------------------------------

    print("=" * 60)
    print("MONITOREO MULTIVARIANTE DE PROCESOS - PCA")
    print("=" * 60)

    rng = default_rng(42)

    print("\nGENERANDO DATOS DEL REACTOR...")


    # Configuración y matriz de covarianza
    params: ReactorParams = configurar_reactor()
    cov_matrix = construir_covarianza(
        params["std_devs"],
        params["correlations"],
    )
    print(params["n_features"], params["n_samples"])

    # Datos de entrenamiento (operación normal)
    X_train = rng.multivariate_normal(
        params["means"],
        cov_matrix,
        size=params["n_samples"]
    )
    print(f"Entrenamiento: {X_train.shape}")

    # --------------------------------------------------------
    # 3.2 Entrenamiento PCA
    # --------------------------------------------------------
    
    modelo = entrenar_pca(X_train, var_threshold=0.90)
    # Cambiar el umbral de varianza para aumentar o disminuir el numero de componentes.

    print(f"\nComponentes seleccionados: {modelo['k']}")

    print(
        f"Varianza explicada acumulada: "
        f"{modelo['var_acum'][modelo['k'] - 1] * 100:.2f}%"
    )

    print(
        f"Límite T² (95%): "
        f"{modelo['T2_limit']:.2f}"
    )

    print(
        f"Límite Q (95%): "
        f"{modelo['Q_limit']:.2f}"
    )

    # --------------------------------------------------------
    # 3.3 Datos de prueba
    # --------------------------------------------------------

    X_test = np.asarray(
        rng.multivariate_normal(
            params["means"],
            cov_matrix,
            size=params["n_samples"]
        ),
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # 3.4 Introducir falla
    # --------------------------------------------------------

    X_test[500:, 1] += 3.5
    print("\nFalla introducida en T2 en 3.5°C a partir de t=500.")

    # --------------------------------------------------------
    # 3.5 Estandarizar datos de prueba
    # --------------------------------------------------------
    mean_train = modelo["mean_train"]
    std_train = modelo["std_train"]
    X_test_scaled = (X_test - mean_train) / std_train

    # --------------------------------------------------------
    # 3.6 Proyección PCA
    # --------------------------------------------------------

    scores_test = (
        X_test_scaled
        @ modelo["V"][:, :modelo["k"]]
    )

    # --------------------------------------------------------
    # 3.7 T²
    # --------------------------------------------------------

    T2_test = np.sum((scores_test ** 2) * modelo["lambda_inv"], axis=1)

    # --------------------------------------------------------
    # 3.8 Q / SPE
    # --------------------------------------------------------

    X_test_reconstructed = (
        scores_test
        @ modelo["V"][:, :modelo["k"]].T
    )
    residuos_test = (X_test_scaled - X_test_reconstructed)
    Q_test = np.sum(residuos_test ** 2, axis=1)

    # --------------------------------------------------------
    # 3.9 Detección de fallas
    # --------------------------------------------------------

    falla = (T2_test > modelo["T2_limit"]) | (Q_test > modelo["Q_limit"])
    fallas_pre: int = int(np.sum(falla[:500]))
    fallas_post: int = int(np.sum(falla[500:]))

    print("\nRESULTADOS")
    print("-" * 40)

    print(f"Falsas alarmas antes de t=500: {fallas_pre}/500")

    print(f"Fallas detectadas después de t=500: {fallas_post}/500")

    # --------------------------------------------------------
    # 3.10 Gráficos
    # --------------------------------------------------------

    #fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    _, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # ========================================================
    # Gráfico T²
    # ========================================================

    axes[0].plot(
        T2_test,
        linewidth=0.8,
        label="Estadístico T²"
    )

    axes[0].axhline(
        y=modelo["T2_limit"],
        linestyle="--",
        linewidth=1.5,
        label=(
            f'Límite 95% '
            f'({modelo["T2_limit"]:.2f})'
        )
    )

    axes[0].axvline(
        x=500,
        linestyle=":",
        linewidth=1,
        label="Inicio de falla"
    )

    axes[0].set_ylabel("T²")

    axes[0].legend(
        loc="upper left"
    )

    axes[0].grid(
        True,
        alpha=0.3
    )

    axes[0].set_title(
        "Monitoreo T² - Distancia dentro del subespacio PCA"
    )

    # ========================================================
    # Gráfico Q
    # ========================================================

    axes[1].plot(
        Q_test,
        linewidth=0.8,
        label="Estadístico Q (SPE)"
    )

    axes[1].axhline(
        y=modelo["Q_limit"],
        linestyle="--",
        linewidth=1.5,
        label=(
            f'Límite 95% '
            f'({modelo["Q_limit"]:.2f})'
        )
    )

    axes[1].axvline(
        x=500,
        linestyle=":",
        linewidth=1,
        label="Inicio de falla"
    )

    axes[1].set_ylabel("Q (SPE)")

    axes[1].legend(
        loc="upper left"
    )

    axes[1].grid(
        True,
        alpha=0.3
    )

    axes[1].set_title(
        "Monitoreo Q - Error de reconstrucción"
    )

    # ========================================================
    # Gráfico de varianza explicada
    # ========================================================

    n_features = params["n_features"]

    axes[2].bar(
        range(1, n_features + 1),
        modelo["var_ratio"],
        alpha=0.6,
        label="Varianza individual"
    )

    axes[2].plot(
        range(1, n_features + 1),
        modelo["var_acum"],
        "ro-",
        label="Varianza acumulada"
    )

    axes[2].axhline(
        y=0.90,
        linestyle="--",
        label="Umbral 90%"
    )

    axes[2].axvline(
        x=modelo["k"],
        linestyle="--",
        label=f'k={modelo["k"]} seleccionado'
    )

    axes[2].set_xlabel(
        "Número de Componente Principal"
    )

    axes[2].set_ylabel(
        "Varianza explicada"
    )

    axes[2].set_title(
        "Varianza explicada - Selección de componentes"
    )

    axes[2].legend(
        loc="lower right"
    )

    axes[2].grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 4. EJECUTAR PROGRAMA
# ============================================================

if __name__ == "__main__":
    main()