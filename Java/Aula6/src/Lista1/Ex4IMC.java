/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Lista1;

import java.util.Scanner;
public class Ex4IMC {
    
    public static void main(String[] args) {
        float IMC, peso, altura;
        Scanner entrada = new Scanner(System.in);
         System.out.println("Bem-vindo,insira seu peso");
         peso = entrada.nextFloat();
         System.out.println("Bem-vindo,insira sua altura");
         altura = entrada.nextFloat();
                 
         IMC = (float) (peso/Math.pow(altura,2));
         System.out.println("Seu IMC = " + IMC);
         
         
        System.out.println("Abaixo de 20 = abaixo do peso.");
        System.out.println("Entre 20 e 25 = dentro do peso.");
        System.out.println("Entre 25 e 30 = Um pouco acima do peso.");
        System.out.println("Entre 30 e 40 = Obesidade grau 1.");
        System.out.println("Maior que 40 = Obesidade mórbida.");
        
    }
  
}
