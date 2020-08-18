/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Lista1;

import java.util.Scanner;

/**
 *
 * @author Breno
 */
public class Ex8 {
    public static void main (String[] args)
  {
    Scanner entrada = new Scanner(System.in);
    System.out.println("Informe as linhas da Matriz: ");
    int dim_mat1 = entrada.nextInt();
      System.out.println("Informe as colunas da matriz: ");
    int dim_mat2 = entrada.nextInt();
    
    if(dim_mat1 != dim_mat2)
    {
      System.out.println("Dimensões divergentes, refaça operação!");
    }
    else
    {
      int [][] numero = new int [dim_mat1][dim_mat2];
 
      for(int i = 0; i < numero.length ; i++)
      {
        for(int j = 0; j < numero.length ; j++)
        {
            if(i==j){
                System.out.print("1");
            } else {
                System.out.print(numero[i][j]);
            }
            
        }
        System.out.println("");
      }
    }
  }
}
