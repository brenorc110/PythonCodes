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
public class Ex5 {
    public static void main(String[] args) {
      int i, h =0;
      Scanner entrada = new Scanner(System.in);
      System.out.println("Bem-vindo,o numero:");
      int n = entrada.nextInt();
       
      for(i=0; i<=n; i++){
      h+=i;
      }
      System.out.println(h);
     }
}
