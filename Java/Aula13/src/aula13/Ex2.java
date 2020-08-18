/*
  Faca um programa que leia uma string e conte quantas consoantes possui o 
 texto.
 */
package aula13;

import java.util.Scanner;

/**
 *
 * @author fsj.5947373464
 */
public class Ex2 {
    public static void main(String[] args) {
        String entrada;
        int a, l, vogais = 0, consoantes = 0;
        Scanner e = new Scanner(System.in);
        System.out.println("Informe o texto: ");
        entrada = e.next();
        
        l = entrada.length();
        
        for (int i=0; i<l; i++){
          char aa = entrada.charAt(i);
          if(aa== 'a'|| aa== 'A' || aa== 'e'|| aa== 'E' || aa== 'i'|| aa== 'I' || aa== 'o'|| aa== 'O' || aa== 'u'|| aa== 'U'){
              vogais++;
          } else consoantes++;
        }
        
        System.out.println("Consoantes da palavra: = " + consoantes);
   
        
    }
    }
