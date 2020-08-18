

import java.util.Scanner;
public class calculaAreaJava {

    public double calcularArea(double b, double h){
        return h*(b/2);
    }
    
    public static void main(String[] args) {
     double b, h, res;
     
      Scanner entrada = new Scanner(System.in);
        System.out.println("Insira a base: ");
	b = entrada.nextDouble();

    System.out.println("Insira a altura: ");
	h = entrada.nextDouble();
        res = b*(h/2);
    System.out.println("Área = " + res);
     
    }
    
}
