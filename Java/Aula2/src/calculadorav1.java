/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */


import java.util.Scanner;

public class calculadorav1 {
    public String op;
    public double res;
    public double n1;
    public double n2;
    
    public double soma(double n1, double n2){
        return n1+n2;
    }
    
    public double subtrair(double n1, double n2){
        return n1-n2;
    }
    
    public double multiplica(double n1, double n2){
        return n1*n2;
    }
    
    public double dividir(double n1, double n2){
        return n1/n2;
    }  
    
    public static void main(String[] args) {
    Scanner entrada = new Scanner(System.in);
    calculadorav1 c1 = new calculadorav1();
    
    System.out.println ("Digite o primeiro n�mero");
    c1.n1 = entrada.nextDouble();
    
    System.out.println ("Digite a opera�o solicitada; 1=Adi��o 2=Subtra��o 3=Mult. 4=Divis�o");
    c1.op = entrada.nextLine();    
    
    System.out.println ("Digite o segundo n�mero");
    c1.n2 = entrada.nextDouble();
    
    //op = entrada.next(); //CONVERTE A ENTRADA PRA STRING
    //opp = Boolean.parseBoolean(op);
    switch (c1.op) {
        case "1":
            System.out.printf ("O Resultado é: "+ c1.soma(c1.n1,c1.n2) + "\n");
            break;
        case "2":
            System.out.printf ("O Resultado é: "+ c1.subtrair(c1.n1,c1.n2) + "\n");
            break;
        case "3":
            System.out.printf ("O Resultado é: "+ c1.multiplica(c1.n1,c1.n2) + "\n");
            break;
        case "4":
            System.out.printf ("O Resultado é: "+ c1.dividir(c1.n1,c1.n2) + "\n");
            break;
        default:
            System.out.printf ("Operação inválida! ");
            break;
        }
    }
}

