entradas = [1, 7, 5]
pesos = [0.8, 0.1, 0]

def soma(x, w):
    soma = 0

    for entrada in range(len(entradas)):
        soma += (x[entrada] * w[entrada])

    return (soma)

def step(soma):
    if (soma >= 1):
        return (1)
    return (0)

print (step(soma(entradas, pesos)))
