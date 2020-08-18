tabuleiro = ['nula','','','','','','','','','']
while True:
    print(tabuleiro)
    simbolo = input('Informe o símbolo do jogador: ')
    posicao=int(input("""
    informe a posição em qua irá jogar:
    1|2|3
    4|5|6
    7|8|9
    """))
    tabuleiro[posicao]= simbolo
    if tabuleiro[1] and tabuleiro[2] and tabuleiro[3] and tabuleiro[4] and tabuleiro[5] and tabuleiro[6] and tabuleiro[7] and tabuleiro[8] and tabuleiro[9]:
        print('Deu velha, abestados!')
        break
    elif tabuleiro[1] == simbolo and tabuleiro[2] == simbolo and tabuleiro[3] == simbolo:
        print('Ganhou', simbolo)
        break
    elif tabuleiro[4] == simbolo and tabuleiro[5] == simbolo and tabuleiro[6] == simbolo:
        print('Ganhou', simbolo)
        break
    elif tabuleiro[7] == simbolo and tabuleiro[8] == simbolo and tabuleiro[9] == simbolo:
        print('Ganhou', simbolo)
        break
    elif tabuleiro[1] == simbolo and tabuleiro[4] == simbolo and tabuleiro[7] == simbolo:
        print('Ganhou', simbolo)
        break
    elif tabuleiro[2] == simbolo and tabuleiro[5] == simbolo and tabuleiro[8] == simbolo:
        print('Ganhou', simbolo)
        break
    elif tabuleiro[3] == simbolo and tabuleiro[6] == simbolo and tabuleiro[9] == simbolo:
        print('Ganhou', simbolo)
        break
    elif tabuleiro[1] == simbolo and tabuleiro[5] == simbolo and tabuleiro[9] == simbolo:
        print('Ganhou', simbolo)
        break
    elif tabuleiro[3] == simbolo and tabuleiro[5] == simbolo and tabuleiro[7] == simbolo:
        print('Ganhou', simbolo)
        break
