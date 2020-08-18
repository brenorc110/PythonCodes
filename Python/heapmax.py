class HeapMaximo:
    def __init__(self, tamanho=11):
        self.vet = [None] * tamanho
        self.tam = 0
        self.ind = 0
        self.num = None
        
    def menu(self):
        txt = """
        Opções:
        1 - Inserir elemento na lista de prioridade
        2 - Consultar o elemento de maior prioridade
        3 - Remover o elemento de maior prioridade
        4 - Consultar toda a lista
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
        print("A lista está vazia")
                

    def opcao1(self):
        if(self.tam < len(self.vet) - 1):
            self.tam += 1
            self.num = int(input("Digite o número a ser inserido na fila: "))
            self.ind = self.tam
            while self.ind > 1 and self.vet[self.pai(self.ind)] < self.num:
                self.vet[self.ind] = self.vet[self.pai(self.ind)]
                self.ind = self.pai(self.ind)
            self.vet[self.ind] = self.num
            print("Número inserido")
        else:
            print("Lista de prioridades lotada")

    def opcao2(self):
        if self.tam == 0:
            print("Lista de prioridades vazia")
        else:
            print("Elemento de maior prioridade: {0}".format(self.vet[1]))

    def opcao3(self):
        if self.tam == 0:
            print("Lista de prioridades vazia")
        else:
            maior_prioridade = self.vet[1]
            self.vet[1] = self.vet[self.tam]
            self.vet[self.tam] = None
            self.tam -= 1
            self.heap_fica(1, self.tam)
            print("O elemento removido: {}".format(maior_prioridade))
        

    def opcao4(self):
        if self.tam == 0:
            print("Lista de prioridades vazia")
        else:
            print("\nTodos os elementos da lista de prioridades:")
            print("\n".join([str(i) for i in self.vet if i is not None]))
            
    
    def pai(self, i):
        return int(i / 2)
    
    def heap_fica(self, i, qtde):
        maior = i
        
        if 2 * i + 1 <= qtde:
            
            f_esq = 2 * i + 1
            f_dir = 2 * i
            if self.vet[f_esq] >= self.vet[f_dir] and self.vet[f_esq] > self.vet[i]:
                maior = 2 * i + 1
            elif self.vet[f_dir] > self.vet[f_esq] and self.vet[f_dir] > self.vet[i]:
                maior = 2 * i
                
        elif 2 * i <= qtde:
            
            f_dir = 2 * i
            if self.vet[f_dir] > self.vet[i]:
                maior = 2 * i
                
        if maior != i:
            aux = self.vet[i]
            self.vet[maior] = aux
            self.heap_fica(maior, qtde)
            
                

    
        
heapMaximo = HeapMaximo()
heapMaximo.main()
