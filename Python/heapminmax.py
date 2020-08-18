import math

class HeapMinMax:
    def __init__(self, tamanho=11):
        self.vet = [None] * tamanho
        self.tam = 0
        self.ind = 0
        self.num = None
        
    def menu(self):
        txt = """
        Opções:
        1 - Inserir elemento na lista de prioridade
        2 - Consultar o elemento de menor prioridade
        3 - Consultar o elemento de maior prioridade
        4 - Remover o elemento de menor prioridade
        5 - Remover o elemento de maior  prioridade
        6 - Consultar toda a lista
        7 - sair
        Digite a sua opção
        """

        opcao = int(input(txt))
        return opcao
    
    def main(self):
        opcoes = {1:self.opcao1, 2:self.opcao2, 3:self.opcao3, 4:self.opcao4, 5:self.opcao5, 6:self.opcao6}
        while True:
            opcao = self.menu()
            if opcao in opcoes:
                opcoes[opcao]()
            else:
                if opcao == 7:
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
            self.inserir_mm(self.num, self.tam)
            print("Número inserido")
        else:
            print("Lista de prioridades lotada")

    def opcao2(self):
        if self.tam == 0:
            print("Lista de prioridades vazia")
        else:
            print("Elemento de menor prioridade: {0}".format(self.vet[1]))
            
    def opcao3(self):
        if self.tam == 0:
            print("Lista de prioridades vazia")
        else:
            mp = self.maior_prior()
            print("Elemento de maior prioridade: {0}".format(self.vet[mp]))

    def opcao4(self):
        if self.tam == 0:
            print("Lista de prioridades vazia")
        else:
            print("O elemento removido: {}".format(self.vet[1]))
            self.vet[1] = self.vet[self.tam]
            maior_prioridade = self.vet[self.tam]
            self.vet[self.tam] = None
            self.tam -= 1
            self.descer(1)           
        

    def opcao5(self):
        if self.tam == 0:
            print("Lista de prioridades vazia")
        elif self.tam == 2:
            print("O elemento removido: {}".format(self.vet[self.tam]))
            self.vet[self.tam] = None
            self.tam = 1
            
        else:
            mmax = 2
            if self.tam >= 3:
                if self.vet[3] > self.vet[2]:
                    mmax = 3
            print("O elemento removido: {}".format(self.vet[mmax]))
            self.vet[mmax] = self.vet[self.tam]
            
            self.vet[self.tam] = None
            self.tam += 1
            self.descer(mmax)
            
            
    def opcao6(self):
        if self.tam == 0:
            print("Lista de prioridades vazia")
        else:
            print("\nTodos os elementos da lista de prioridades:")
            print("\n".join([str(i) for i in self.vet if i is not None]))
            
    
    def inserir_mm(self, num, i):
        self.vet[i] = num
        self.subir(i)
        
    def subir(self, i):
        pai = int(i / 2)
        if self.minimo(i) is True:
            if pai >= 1:
                if self.vet[i] > self.vet[pai]:
                    self.trocar(i, pai)
                    self.subir_max(pai)
                else:
                    self.subir_min(i)
        else:
            if pai >= 1:
                if self.vet[i] < self.vet[pai]:
                    self.trocar(i, pai)
                    self.subir_min(pai)
                else:
                    self.subir_max(i)
    
    def subir_min(self, i):
        avo =  int(i / 4)
        if avo >= 1 and self.vet[i] < self.vet[avo]:
            self.trocar(i, avo)
            self.subir_min(avo)
            
    def subir_max(self, i):
        avo = int(i / 4)
        if avo >= 1 and self.vet[i] > self.vet[avo]:
            self.trocar(i, avo)
            self.subir_max(avo)
        
    def maior_prior(self):
        if self.tam == 1:
            return 1
        elif self.tam > 2:
            if self.vet[3] is not None and self.vet[3] > self.vet[2]:
                return 3
            else:
                if self.vet[2] is not None:
                    return 2

        else:
            return 2
        
    def descer(self, i):
        if self.minimo(i) is True:
            self.descer_min(i)
        else:
            self.descer_max(i)
            
    def minimo(self, i):
        nivel = int(math.log2(i)) + 1
        if nivel % 2 == 0:
            return False
        else:
            return True
        
    def descer_min(self, i):
        if 2 * i <= self.tam:
            m = self.min_descendente(i)
            if self.vet[i] > self.vet[m]:
                self.trocar(i, m)
                if m >= 4 * i:
                    p = int(m / 2)
                    if self.vet[p] < self.vet[m]:
                        self.trocar(p, m)
                    self.descer_min(m)
                    
    def min_descendente(self, i):
        m = 0
        if 2 * i <= self.tam:
            m = 2 * i
            if self.vet[m+1] is not None and self.vet[m+1] < self.vet[m]:
                m = m+1
            k = 4 * i
            while k <= 4 *i +3 and k <= self.tam:
                if self.vet[k] is not None and self.vet[m] is not None:
                    if self.vet[k] < self.vet[m]:
                        m = k
                k += 1
        return m
    
    def descer_max(self, i):
        if 2 * i <= self.tam:
            m = self.max_descendente(i)
            if self.vet[m] is not None:
                if self.vet[i] < self.vet[m]:
                    self.trocar(i, m)
                
                    if m >= 4 * i:
                        p = int(m / 2)
                        if self.vet[p] > self.vet[m]:
                            self.trocar(p, m)
                        self.descer_max(m)
                    
    def max_descendente(self, i):
        m = 0
        if 2 * i <= self.tam:
            m = 2 * i
            if self.vet[m] is not None and self.vet[m+1] is not None:
                if self.vet[m+1] > self.vet[m]:
                    k = 4 * i
                    while k <= 4 *i + 3 and k <= self.tam:
                        if self.vet[k] > self.vet[m]:
                            m = k
                        k+=1
        return m
    
    def trocar(self, x, y):
        self.vet[x], self.vet[y] = self.vet[y], self.vet[x]
                
                

    
        
heapMinMax = HeapMinMax()
heapMinMax.main()
