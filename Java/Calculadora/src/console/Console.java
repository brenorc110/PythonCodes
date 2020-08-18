package console;

import java.util.Scanner;

public class Console {
    public String op;
    public double res;
    public double n1;
    public double n2;
    
    public void calcula(double n1, String op, double n2){
        if (op.equals("-")){
            System.out.println("O Resultado é: " + (n1-n2));
        }else if (op.equals("+")){
            System.out.println("O Resultado é: " + (n1+n2));
        }else if (op.equals("*")){
            System.out.println("O Resultado é: " + (n1*n2));
        }else if (op.equals("/")){
            System.out.println("O Resultado é: " + (n1/n2));
        }else if (op.equals("pot")){
            System.out.println("O Resultado é: " + (Math.pow(n1,n2)));
        }else if (op.equals("raiz")){
            System.out.println("O Resultado é: " + (Math.pow(n1,1.0/n2)));
        }
    }  
    
    public static void main(String[] args) {
    Scanner entrada = new Scanner(System.in);
    Console c1 = new Console();
    
    System.out.println ("Digite o primeiro número");
    c1.n1 = entrada.nextDouble();
    
    System.out.println ("Entre com o operador: +,-,*,/,pot,raiz");
    c1.op = entrada.next();    
    
    System.out.println ("Digite o segundo número");
    c1.n2 = entrada.nextDouble();
    
    c1.calcula(c1.n1, c1.op, c1.n2); 
    entrada.close();   
    }
    

	
	

	}

