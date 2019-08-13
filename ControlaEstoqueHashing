tam = 10
class hash:
    produto = None
    chave = None
    livre = "L"
    
class Controle:
    def __init__(self, tam=8):
        self.tam = tam
        self.tabela = [hash() for _ in range(self.tam)]
     
    #Realiza o hash com o Codigo ASCII do caractere
    def funcao_hashing(self, num):
        return num % self.tam
    
    #Busca através do caractere o produto na lista
    def mostrar_hash(self, n):   
        for i in range(self.tam):
           if self.tabela[i].chave == n:
               print(self.tabela[i].produto)
           else:
               print("Não Há produtos cadastrados deste tipo")
               break
                
    def inserir(self,pr,pos, n):
        i = 0
        while i < self.tam:
            print()
            if self.tabela[(pos+i)%self.tam].livre != "L" \
            and self.tabela[(pos+i)%self.tam].livre != "R":
                i = i + 1
            else:
                if i < self.tam:
                    self.tabela[(pos +i) % self.tam].produto = pr
                    self.tabela[(pos +i) % self.tam].chave = n
                    self.tabela[(pos +i) % self.tam].livre = "O"
                    break
                else:
                    print("Tabela está cheia")
                    
 
    #Busca Tipo através de um Submenu, e percorre um dicionário de tipos aceitos    
    def opcao1(self):
        esc = self.submenu()
        pd={"a":1,"h":2,"l":3,"v":4}
        if esc in pd:
            pos = self.funcao_hashing(ord(esc))
            n = input("Digite O Produto: ")
            self.inserir(n,pos, esc)
        else:
            self.opcao1()
    
    #Conta tendo o caractere como entrada quantos registros há
    def buscar(self, n):   
        cont=0
        for i in range(self.tam):
           if self.tabela[i].chave == n:
               cont = cont +1
        return cont

    # Busca especificamente de um tipo
    def opcao2(self):
        n = self.submenu()
        self.mostrar_hash(n)

    #Opção 3 chama abusca para contar quantos há
    def opcao3(self):
       print("Há {} Produtos de Alimentacao".format(self.buscar("a")))
       print("Há {} Produtos de Higiene".format(self.buscar("h")))
       print("Há {} Produtos de Limpeza".format(self.buscar("l")))
       print("Há {} Produtos de Vestuario".format(self.buscar("v")))
    
    def menu(self):
        txt = """
        Opções:
            1-Inserir Produto;
            2-Consultar todos os produtos cadastrados de um tipo;
            3-Contar quantos Estão Cadastrados em cada Tipo
            4-sair;
        Digite a sua opção
        """
        opcao = int(input(txt))
        return opcao
   
    #Submenu de tipos de produtos para cadastrar
    def submenu(self):
        txt1 = """
        Tipo:
            (A)limentacão;
            (H)igiene;
            (L)impeza;
            (V)estuário;
        Digite a sua opção
        """
        Prd = input(txt1)
        return Prd
    
    def main(self):
        opcoes = {1:self.opcao1, 2:self.opcao2, 3:self.opcao3}
        while True:
            opcao = self.menu()
            if opcao in opcoes:
                opcoes[opcao]()
            else:
                if opcao == 4:
                    break
                else:
                    print("Opção inválida")
                    
controle = Controle()
controle.main()
