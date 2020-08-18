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
public class Ex3 {
    
     public static void main(String[] args) {
      Scanner entrada = new Scanner(System.in);
      System.out.println("Bem-vindo,insira sua idade:");
      int id = entrada.nextInt();
      String categ;
      if(id<=7) categ="Infantil A"; else if(id<=11 && id>=8) categ="Infantil B"; else if (id==12 || id==13) categ="Juvenil A"; else if(id>=14 && id<=17) categ="Juvenil B"; else categ="Adulto";
         System.out.println("Categoria: "+ categ);
     }
        
        
}
