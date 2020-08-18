/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package prova02;

import java.util.Scanner;

public class validaCPF {

	public static void main(String[] args) 
	{
		int soma1, soma2; 
		float resto1, resto2;
		
		Scanner e = new Scanner(System.in);
        System.out.println("Informe o CPF: ");
        String CPF = e.next();
        
		char[] temp = CPF.toCharArray();
		int[] num = new int[temp.length];
		
		for (int i = 0; i < temp.length; i++) 
		{
			num[i] = Integer
					.parseInt(String.valueOf(temp[i]));
					//System.out.println(num[i]);
		}
		
		System.out.println(num[0]);
		System.out.println(num[10]);
		
		if (num[0]==num[1] && num[1]==num[2] && num[2]==num[3] && num[3]==num[4] && num[4]==num[5] && num[5]==num[6] && num[6]==num[7] && num[7]==num[8] && num[8]==num[9] || num.length!=11){
            System.out.println("CPF irregular");
        }else{
            soma1 = num[0] * 10 + num[1] * 9 + num[2] * 8 + num[3] * 7 + num[4] * 6 + num[5] * 5 + num[6] * 4 + num[7] * 3 + num[8] * 2;
            resto1 = soma1%11;
            System.out.println("Resto1= " + resto1);
            if(resto1<2){
                resto1=0;
            }else{
                resto1= 11- resto1;
            }
            
            soma2= num[0] * 11 + num[1] * 10 + num[2] * 9 + num[3] * 8 + num[4] * 7 + num[5] * 6 + num[6] * 5 + num[7] * 4 + num[8] * 3 + num[9] * 2;
            resto2 = soma2%11;
            System.out.println("Resto2= " + resto2);
            if(resto2<2){
                resto2=0;
            } else{
                resto2= 11- resto2;
            }
            
            if(resto1==num[9] && resto2==num[10]){
                System.out.println("OK, CPF válido! ");
            }else{
                System.out.println("CPF inválido!");
            }
            //http://www.macoratti.net/alg_cpf.htm
        }
	}
	
}
