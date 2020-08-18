def main():
    cedatual = 100
    cedqt = 0
    ced100 = 0
    ced50 = 0
    ced20 = 0
    ced10 = 0
    ced5 = 20
    totalcaixa = (ced100*100) + (ced50*50) + (ced20*20) + (ced10*10) + (ced5*5)
    contSaque = 'S'
    
    while contSaque== 'S':
        saque = int(input("Digite um valor para sacar:\n"))
        total = saque
        while True:
            if saque <= totalcaixa:
                if total >= cedatual:
                        total -= cedatual#200rs-100 = 100rs
                        cedqt += 1 #notas +1
                else:
                    if cedatual == 100 and ced100==0:
                        cedatual = 50
                    elif cedatual == 100 and ced100>0:
                        cedatual = 50
                        ced100 = ced100 - cedqt
                        print("É necessário",cedqt,"cedulas de",cedatual)
                        print("Ejetando nota de 100")
                    elif cedatual == 50 and ced50==0:
                        cedatual = 20
                    elif cedatual == 50 and ced50>0:
                        cedatual = 20
                        ced50 = ced50 - cedqt
                        print("É necessário",cedqt,"cedulas de",cedatual)
                        print("Ejetando nota de 50")
                    elif cedatual == 20 and ced20==0:
                        cedatual = 10
                    elif cedatual == 20 and ced20>0:
                        cedatual = 10
                        ced20 = ced20 - cedqt
                        print("É necessário",cedqt,"cedulas de",cedatual)
                        print("Ejetando nota de 20")
                    elif cedatual == 10 and ced10==0:
                        cedatual = 5
                    elif cedatual == 10 and ced10>0:
                        cedatual = 5
                        ced10 = ced10 - cedqt
                        print("É necessário",cedqt,"cedulas de",cedatual)
                        print("Ejetando nota de 10")
                    elif cedatual == 5 and ced5==0:
                        cedatual = 50
                    elif cedatual == 5 and ced5>0:
                        cedatual = 50
                        ced5 = ced5 - cedqt
                        print("É necessário",cedqt,"cedulas de",cedatual)
                        print("Ejetando nota de 5")
                        
                    if total > 0 and total <5:
                        print("O restante do valor não pode ser pago:",total)
                        break
                    if total == 0:
                        print("\nAinda restam",ced100,"cedulas de 100")
                        print("Ainda restam",ced50,"cedulas de 50")
                        print("Ainda restam",ced20,"cedulas de 20")
                        print("Ainda restam",ced10,"cedulas de 10")
                        print("Ainda restam",ced5,"cedulas de 5\n")
                        break
        else:
            print("Este valor excede a capacidade deste terminal.")
        contSaque = input("Deseja continuar as operações de saque?: (S/N)\n")
        
        
if __name__ == "__main__":
    main()
