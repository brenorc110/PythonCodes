package Lista1;

import java.util.Scanner;

public class Ex2 {
    public static void main(String[] args) {
     Scanner entrada = new Scanner(System.in);
     double peso, total;
        System.out.println("Bem-vindo, insira o peso do prato em KG: ");
        peso = entrada.nextDouble();
        total = 12*(peso-0.100);
        
        System.out.println("Total a pagar: R$ " + total);
    }
    
}
