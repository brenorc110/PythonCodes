class Fila:
    num = None;
    prox = None
    def __del__(self):
        print("Objeto será deletado")
    
class Controle:
    
    def __init__(self):
        self.inicio = None
        self.fim = None
        self.aux = None
    
    def menu(self):
        txt = """
        Opções:
        1 - Inserir na fila
        2 - Consultar toda a fila
        3 - Remover da fila
        4 - Esvaziar a fila
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
        print("A fila está vazia")
                

    def opcao1(self):
        novo = Fila()
        novo.num = int(input("Digite o número a ser inserido na fila: "))
        novo.prox = None
        if self.inicio is None:
            self.inicio = novo
            self.fim = novo
        else:
            self.fim.prox = novo
            self.fim = novo

    def opcao2(self):
        if self.inicio is None:
            self.printpilhavazia()
        else:
            print("Consultando a fila!")
            self.aux = self.inicio
            while self.aux is not None:
                print(self.aux.num)
                self.aux = self.aux.prox

    def opcao3(self):
        if self.inicio is None:
            self.printpilhavazia()
        else:
            print("A fila contém elementos e o último elemento inserido será removido")
            self.inicio = self.inicio.prox           
        

    def opcao4(self):
        if self.inicio is None:
            self.printpilhavazia()
        else:
            print("A fila será esvaziada")
            # Usar essa opção mantém lixo na memória
            # self.topo = None 
            
            self.aux = self.inicio
            while self.aux is not None:
                self.inicio = self.inicio.prox
                del self.aux
                self.aux = self.inicio
                
            print("fila esvaziada")

    
        
controle = Controle()
controle.main()