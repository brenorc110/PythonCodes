import java.util.Scanner;

public class calculadorav2 {
    public String op;
    public double res;
    public double n1;
    public double n2;
    
    public void calcula(double n1, String op, double n2){
        if (op.equals("-")){
            System.out.println("O Resultado �: " + (n1-n2));
        }
        if (op.equals("+")){
            System.out.println("O Resultado �: " + (n1+n2));
        }
        if (op.equals("*")){
            System.out.println("O Resultado �: " + (n1*n2));
        }
        if (op.equals("/")){
            System.out.println("O Resultado �: " + (n1/n2));
        }
		if (op.equals("pot")){
            System.out.println("O Resultado �: " + (Math.pow(n1,n2)));
        }
		if (op.equals("raiz")){
            System.out.println("O Resultado �: " + (Math.pow(n1,1.0/n2)));
        }
    }  
    
    public static void main(String[] args) {
    Scanner entrada = new Scanner(System.in);
    calculadorav2 c1 = new calculadorav2();
    
    System.out.println ("Digite o primeiro n�mero");
    c1.n1 = entrada.nextDouble();
    
    System.out.println ("Entre com o operador: +,-,*,/,pot,raiz");
    c1.op = entrada.next();    
    
    System.out.println ("Digite o segundo n�mero");
    c1.n2 = entrada.nextDouble();
    
    c1.calcula(c1.n1, c1.op, c1.n2);    
    }

	
	

	}

