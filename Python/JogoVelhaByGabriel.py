def main():
    velha="""               Posições
   |   |      7 | 8 | 9
---+---+---  ---+---+---
   |   |      4 | 5 | 6
---+---+---  ---+---+---
   |   |      1 | 2 | 3
    """
    posições = [
           None,  
          (5, 1), 
          (5, 5), 
          (5, 9), 
          (3, 1), 
          (3, 5), 
          (3, 9), 
          (1, 1),  
          (1, 5), 
          (1, 9),  
        ]

    ganho = [
              [ 1, 2, 3], 
              [ 4, 5, 6],
              [ 7, 8, 9],
              [ 7, 4, 1], 
              [ 8, 5, 2],
              [ 9, 6, 3],
              [ 7, 5, 3], 
              [ 1, 5, 9]
            ]

    tabuleiro = []
    for linha in velha.splitlines():
        tabuleiro.append(list(linha))

    jogador = "X" 
    jogando = True
    jogadas = 0 
    while True:
        for t in tabuleiro: 
            print("".join(t))
        if not jogando: 
            break
        if jogadas == 9: 
            print("Deu velha! Ninguém ganhou.")
            break
        jogada = int(input("Digite a posição a jogar 1-9 (jogador %s):" % jogador))
        if jogada<1 or jogada>9:
            print("Posição inválida")
            continue
        if tabuleiro[posições[jogada][0]][posições[jogada][1]] != " ":
            print("Posição ocupada.");
            continue
        tabuleiro[posições[jogada][0]][posições[jogada][1]] = jogador
        for p in ganho:
            for x in p:
                if tabuleiro[posições[x][0]][posições[x][1]] != jogador:
                    break
            else: 
                print("O jogador %s ganhou (%s): "%(jogador, p))
                jogando = False
                break
        jogador = "X" if jogador == "O" else "O" 
        jogadas +=1 

if __name__ == '__main__':
    main()
