package ArvoreBinaria;

import java.util.Scanner;

/**
 * Hello world!
 *
 */
public class No 
{
    int valor;
    //Apontadores do tipo nó
    No noEsquerda;
    No noDireita;

    public No(int valor){
        //Cria o no
        this.valor = valor;
    }

    public static No raiz;

    public static void inserir(int valor) {
        inserir(raiz, valor);
    }
    private static void inserir(No node, int valor) {
        //Percorre pela raiz
        if(node==null){
            //Arvore vazia
            raiz = new No(valor);
        } else{
            //Se arvore ja tiver algum elemento, avalia se valor menor que o do nó corrente
            if(valor<node.valor){
                if(node.noEsquerda !=  null){
                    //Volta a percorrer metodo recursivamente
                    inserir(node.noEsquerda, valor);
                }else{
                    System.out.println( "Inserindo " + valor + " a esquerda de " + node.valor);
                    node.noEsquerda = new No(valor);
                }
            } else{
                if(node.noDireita !=  null){
                    inserir(node.noDireita, valor);
                }else{
                    System.out.println( "Inserindo " + valor + " a direita de " + node.valor);
                    node.noDireita = new No(valor);
                }

            }
        }
    }

    public static void preordem(No node) {
        if (node!=null){
            System.out.println(node.valor + ", ");
            preordem(node.noEsquerda);
            preordem(node.noDireita);
        }
    }
    public static void posordem(No node) {
        if(node!=null){
        posordem(node.noEsquerda);
        posordem(node.noDireita);
        System.out.println(node.valor + ", ");
        }
    }
    public static void ordem(No node) {
        if (node!=null){
            ordem(node.noEsquerda);
            System.out.println(node.valor + ", ");
            ordem(node.noDireita);
        }
    }
    public static void main( String[] args )
    {   
        int i =0, tam=0;
        Scanner ent = new Scanner(System.in);
        System.out.println("Quantos valores quer inserir?");
        tam = ent.nextInt();
        do{
            System.out.println("Insira um número inteiro: ");
            int num = ent.nextInt();
            inserir(num);
            i++;
        } while (i<tam);

        /*
        inserir(10);
        inserir(5);
        inserir(15);
        inserir(2);
        inserir(4);
        */
        System.out.println("Percorrendo arvore...");
        System.out.println("\nPre-ordem: ");
        preordem(raiz);
        System.out.println("\nPos-ordem: ");
        posordem(raiz);
        System.out.println("\nOrdem: ");
        ordem(raiz);
        System.out.println("");
    }
}
