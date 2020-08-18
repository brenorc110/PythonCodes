/*
    leia um texto e conte quantas frases podem ser encontradas. Considere , e
 . como identificadores de frases. 
 */
package aula13;

import java.util.Scanner;

/**
 *
 * @author fsj.5947373464
 */
public class Ex3 {
    public static void main(String[] args) {
        String entrada;
        int a, frases = 0;
        Scanner e = new Scanner(System.in);
        System.out.println("Informe o texto: ");
        entrada = e.next();
        
        for (int i=0; i<entrada.length(); i++){
            
            if((entrada.charAt(i) == ',') || (entrada.charAt(i) == '.')){
            frases++;
            }
        }
        System.out.println("Frases= "+ frases);
    }
    
}
