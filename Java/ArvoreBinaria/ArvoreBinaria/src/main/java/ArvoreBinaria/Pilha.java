package ArvoreBinaria;

import java.util.Scanner;

public class Pilha{
    int elementos[];
    int topo;

    public Pilha(int tamanho) {
        elementos = new int[tamanho];
        topo = -1; //Recebe uma posicao negativa, invalida
    }

    public void push(int e) {
        if (isFull()) {
            throw new RuntimeException("Stack overflow!!!");
        }
        topo++;
        elementos[topo] = e;
    }

    public int pop() {
        if (isEmpty()) {
            throw new RuntimeException("Empty stack!!!");
        }
        int e;
        e = elementos[topo];
        topo--;
        return e;
    }

    public boolean isEmpty() {
        return (topo == -1);
    }

    public boolean isFull() {
        return (topo == (elementos.length) - 1);
    }

    public int top() {
        if (isEmpty()) {
            throw new RuntimeException("Empty stack!!!");
        }
        return (elementos[topo]);
    }

    public void desempilhar(Pilha p) {
        while (!p.isEmpty()) {
            int elemento = p.pop();
            System.out.println("Desempilhado: " + elemento);
        }
    }

    public static void main(String[] args) {
        int i = 0;
        Scanner ent = new Scanner(System.in);
        System.out.println("Insira o tamanho da pilha: ");
        int tamanho = ent.nextInt();
        Pilha p = new Pilha(tamanho);

        do {
            System.out.println("Insira elemento: ");
            int e = ent.nextInt();
            p.push(e);
            i++;
        }while(i<tamanho); 
        p.desempilhar(p);    
    }
        
}