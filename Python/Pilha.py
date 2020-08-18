class Pilha:
    num = None;
    prox = None
    def __del__(self):
        print("Objeto será deletado")
    
class Controle:
    
    def __init__(self):
        self.topo = None
        self.aux = None
    
    def menu(self):
        txt = """
        Opções:
        1 - Inserir na pilha
        2 - Consultar toda a pilha
        3 - Remover da pilha
        4 - Esvaziar a pilha
        5 - sair
        Digite a sua opção
        """

        opcao = int(input(txt))
        return opcao
    
    def main(self):
        opcoes = {1:self.opcao1, 2:self.opcao2, 3:self.opcao3, 4:self.opcao4}
        while True:
            opcao = self.menu()
            if opcao in opcoes:
                opcoes[opcao]()
            else:
                if opcao == 5:
                    break
                else:
                    print("Opção inválida")
                                
    def printpilhavazia(self):
        print("A pilha está vazia")
                

    def opcao1(self):
        novo = Pilha()
        novo.num = int(input("Digite o número a ser inserido na pilha: "))
        novo.prox = self.topo
        self.topo = novo

    def opcao2(self):
        if self.topo is None:
            self.printpilhavazia()
        else:
            print("Consultando a pilha!")
            self.aux = self.topo
            while self.aux is not None:
                print(self.aux.num)
                self.aux = self.aux.prox

    def opcao3(self):
        if self.topo is None:
            self.printpilhavazia()
        else:
            print("A pilha contém elementos e o último elemento inserido será removido")
            self.topo = self.topo.prox
            
        

    def opcao4(self):
        if self.topo is None:
            self.printpilhavazia()
        else:
            print("A pilha será esvaziada")
            # Usar essa opção mantém lixo na memória
            # self.topo = None 
            
            self.aux = self.topo
            while self.aux is not None:
                self.topo = self.topo.prox
                del self.aux
                self.aux = self.topo
                
            print("Pilha esvaziada")

    
        
controle = Controle()
controle.main()