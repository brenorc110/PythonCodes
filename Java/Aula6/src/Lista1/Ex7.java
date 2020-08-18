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

public class Ex7 {
    public static void main(String[] args) {
        int i, c, h;
         Scanner entrada = new Scanner(System.in);
         System.out.println("Bem-vindo,o numero:");
        c = entrada.nextInt();
        System.out.println("Resultado: ");
        for(i=1; i<=c; i++){
        for(h=i; h<c+i; h++){
            System.out.print(h);
        }
            System.out.println("");
        }
    }
}

