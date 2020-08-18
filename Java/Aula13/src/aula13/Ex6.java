/*
    Conte quantas silabas possuem em um texto (Silaba=junção de vogal e 
 consoante). 
 */
package aula13;

/**
 *
 * @author fsj.5947373464
 */
import java.util.Scanner;

/**
 *
 * @author fsj.5947373464
 */
public class Ex6 {
    public static void main(String[] args) {
        String entrada;
        int i, tamanho, silabas = 0;
        Scanner e = new Scanner(System.in);
        System.out.println("Informe o texto: ");
        entrada = e.next();
        
        tamanho = entrada.length();
        for (i=0; i<tamanho-1; i++){
         if(entrada.charAt(i)!= 'a'&& entrada.charAt(i)!= 'A' &&
                 entrada.charAt(i)!= 'e' && entrada.charAt(i)!= 'E' &&
                 entrada.charAt(i)!= 'i'&& entrada.charAt(i)!= 'I' &&
                 entrada.charAt(i)!= 'o'&& entrada.charAt(i)!= 'O' &&
                 entrada.charAt(i)!= 'u'&& entrada.charAt(i)!= 'U' &&
                 entrada.charAt(i+1)=='a'|| entrada.charAt(i+1)=='A' ||
                 entrada.charAt(i+1)=='e'|| entrada.charAt(i+1)=='E' ||
                 entrada.charAt(i+1)=='i'|| entrada.charAt(i+1)=='I' ||
                 entrada.charAt(i+1)=='o'|| entrada.charAt(i+1)=='O' || 
                 entrada.charAt(i+1)=='u'|| entrada.charAt(i+1)=='U'){
          
              silabas++;
          }
        }
        
        System.out.println("Silabas da palavra= " + silabas);
    }
    }
