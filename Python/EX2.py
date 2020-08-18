#usr/bin/env python
# -*- coding: utf-8 -*-
class LinkedList(object):
    def __init__(self, tipo, descicao):
        self.root = None
        self.size = None
        self.tipo = tipo
        self.h = h
        self.tam = tam
        self.descricao = descricao
    def funcao_hashing(descricao, tam):
        h = '%s: %s'.__mod__((descricao,tam))
        return h

    #descricao = Elemento que será operado. 
    def add(self, h, descricao):
        if (lista[h]is None):
            lista[h] = descricao
        else:
          while True:
            h = h+1
            if (lista[h]is None):
              lista[h] = descricao
              break
        if self.root:
            #insercao, quando lista possui elementos. 
            pointer = self.root
            while(pointer.next):
                pointer = pointer.next
            pointer.next = Controle(descricao)
        else:
            # primeira inserção: 
            self.root = Controle(descricao)
        self.size = self.size + 1
    def __getitem__(self, index):
        return lista[h]
    
    def busca(self, tipo):
        i = 0
        while(pointer):
            if pointer.data == tipo:
                return Controle(descricao)
            pointer = pointer.next
            i = i+1
        raise ValueError("Elemento não está na lista. ".format(tipo))
    
class Controle(object):
  root = None
  def menu(self):
        txt ="""
    Opções:
     	1-Inserir produto;
     	2-Exibir itens dos produtos cadastrados por tipo;
     	3-Exibir quantos produtos cadastrados por tipo;
     	4-Sair.
    
      Digite a sua opção:
            """
        opcao = int(input(txt))
        return opcao
  
  def main (self):
    opcoes = {1:self.opcao1, 2:self.opcao2, 3:self.opcao3}
    while True:
      opcao = self.menu()
      if opcao in opcoes:
        opcoes[opcao]()
      elif opcao == 4:
          print("Saindo... ")
          break
      else:
          print("Opcão Inválida")
          break

  def opcao1(self):
        tipo = (input("Insira o tipo do produto: A - Alimentação, H - Higiene, L - Limpeza, V - Vestuário "))
        descricao = (input("Insira sua descrição: "))
        LinkedList.funcao_hashing(descricao, 1)
        LinkedList.add(h, descricao)

  def opcao2(self):
        tipo = (input("Insira o tipo do produto a ser consultado: "))
        print("Temos %d", lista.count(tipo), " elementos do tipo no estoque.")

  def opcao3(self):
        tipo = (input("Insira o tipo do produto a ser consultado: "))
        print("Temos %d", lista.count("a"), " elementos do tipo ALIMENTO no estoque.")
        print("Temos %d", lista.count("h"), " elementos do tipo HIGIENE no estoque.")
        print("Temos %d", lista.count("l"), " elementos do tipo LIMPEZA no estoque.")
        print("Temos %d", lista.count("v"), " elementos do tipo VESTUÁRIO no estoque.")

controle = Controle()
controle.main()
