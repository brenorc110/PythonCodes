/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Lista1;

import java.util.Scanner;

/**
 *
 * @author fsj.5947373464
 */
public class Ex6 {

    public static void main(String[] args) {
        int i = 0, n;
        String e="s";
        while(e.equalsIgnoreCase("s")){
            Scanner entrada = new Scanner(System.in);
            System.out.println("Bem-vindo,o numero:");
            n = entrada.nextInt();
            if (n %2==0) System.out.println("Numero par.");
            else System.out.println("Número impar");
              System.out.println("Gostaria de informar outro número? S/N");
              e = entrada.next();
        }
        
    }

}
