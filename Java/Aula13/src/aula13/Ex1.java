/*
  Faca um programa que leia uma string e conte separadamente 
 a quantidade de cada vogal.
 */
package aula13;

import java.util.Scanner;

/**
 *
 * @author fsj.5947373464
 */
public class Ex1 {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        String entrada;
        int a, l, as = 0, es=0 , is=0, os=0, us=0;
        Scanner e = new Scanner(System.in);
        System.out.println("Informe o texto: ");
        entrada = e.next();
        
        l = entrada.length();
        
        for (int i=0; i<l; i++){
          char aa = entrada.charAt(i);
          if(aa== 'a'|| aa== 'A'){
              as++;
          } 
          if(aa== 'e'|| aa== 'E'){
              es++;
          }
          if(aa== 'i'|| aa== 'I'){
              is++;
          }
          if(aa== 'o'|| aa== 'O'){
              os++;
          }
          if(aa== 'u'|| aa== 'U'){
              us++;
          }
        }
        System.out.println("As = " + as);
        System.out.println("Es = " + es);
        System.out.println("Is = " + is);
        System.out.println("Os = " + os);
        System.out.println("Us = " + us);
        
    }
    
}
