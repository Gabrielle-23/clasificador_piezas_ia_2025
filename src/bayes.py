# src/bayes/estimator.py

from collections import Counter

# 1) Definimos tipos de piezas
PIEZAS = ["tornillo", "clavo", "arandela", "tuerca"]

# 2) Probabilidades P(pieza | caja)
#    (cambiá estos valores si tu enunciado tiene otros)
LIKELIHOODS = {
    "A": {"tornillo": 0.25, "clavo": 0.25, "arandela": 0.25, "tuerca": 0.25},
    "B": {"tornillo": 0.15, "clavo": 0.30, "arandela": 0.30, "tuerca": 0.25},
    "C": {"tornillo": 0.25, "clavo": 0.35, "arandela": 0.25, "tuerca": 0.15},
    "D": {"tornillo": 0.50, "clavo": 0.50, "arandela": 0.00, "tuerca": 0.00},
}

# 3) Prior inicial P(caja)
def prior_uniforme():
    n = len(LIKELIHOODS)
    return {caja: 1.0 / n for caja in LIKELIHOODS.keys()}

def normalizar(distrib):
    total = sum(distrib.values())
    if total == 0:
        raise ValueError("Probabilidades suman 0, algo está mal en los likelihoods o en la muestra.")
    return {k: v / total for k, v in distrib.items()}

# 4) Actualizar con UNA pieza observada
def actualizar_con_pieza(prior, pieza):
    """Aplica Bayes para una observación 'pieza'."""
    posterior_sin_norm = {}
    for caja, p_caja in prior.items():
        p_pieza_dado_caja = LIKELIHOODS[caja][pieza]
        posterior_sin_norm[caja] = p_pieza_dado_caja * p_caja
    posterior = normalizar(posterior_sin_norm)
    return posterior

# 5) Actualizar secuencialmente con toda la muestra (lista de piezas)
def estimar_posterior_secuencial(muestra):
    """
    muestra: lista como ['tornillo', 'tornillo', 'clavo', ...]
    Devuelve:
        posterior_final, historial (lista de distribuciones después de cada pieza)
    """
    prior = prior_uniforme()
    historial = [prior]
    for pieza in muestra:
        prior = actualizar_con_pieza(prior, pieza)
        historial.append(prior)
    return prior, historial

# 6) Versión alternativa "por conteo" (todo junto, sin orden)
def estimar_posterior_por_conteo(muestra):
    """
    Usa solo los conteos de cada tipo de pieza.
    Resultado final es el mismo que la versión secuencial.
    """
    conteos = Counter(muestra)  # ej: {'tornillo': 7, 'clavo': 2, 'arandela': 1}
    prior = prior_uniforme()

    numeradores = {}
    for caja, p_caja in prior.items():
        # calculamos P(muestra | caja)
        verosimilitud = 1.0
        for pieza, n in conteos.items():
            verosimilitud *= LIKELIHOODS[caja][pieza] ** n
        numeradores[caja] = verosimilitud * p_caja

    posterior = normalizar(numeradores)
    return posterior

# 7) Ejemplo de uso simple (podés cambiar la muestra)
if __name__ == "__main__":
    # Ejemplo: 10 piezas clasificadas por K-Means
    muestra = [
        "tornillo", "tornillo", "clavo", "tornillo", "arandela",
        "tornillo", "tornillo", "clavo", "tornillo", "tornillo"
    ]

    posterior_final, historial = estimar_posterior_secuencial(muestra)

    print("Probabilidad de cada caja (estimator secuencial):")
    for caja, p in posterior_final.items():
        print(f"  Caja {caja}: {p:.3f}")

    # También podemos comparar con el cálculo por conteo
    posterior_conteo = estimar_posterior_por_conteo(muestra)
    print("\nProbabilidades (versión por conteo):")
    for caja, p in posterior_conteo.items():
        print(f"  Caja {caja}: {p:.3f}")
