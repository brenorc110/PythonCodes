/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Lista2;

import java.util.Scanner;

/**
 *
 * @author Breno
 */
public class Cliente {
    Rg rg;
    Endereco end;
    void Cliente(){
        System.out.println("Aqui temos a classe cliente sendo executada.");  
    }
    void alterarEndereco(){
    }
    
    public static void main(String[] args) {
        Endereco end = new Endereco();
        Scanner entrada = new Scanner(System.in);
        System.out.println("Insira o novo logradouro: ");
        end.setLogradouro(entrada.next());
        
        System.out.println("Insira a nova cidade: ");
        end.setCidade(entrada.next());
        
        System.out.println("Insira o novo bairro: ");
        end.setBairro(entrada.next());
        
        System.out.println("Insira o novo estado: ");
        end.setEstado(entrada.next());
        
        System.out.println("Insira o novo país: ");
        end.setPais(entrada.next());
        
        System.out.println("Insira o novo número: ");
        end.setNumero(entrada.nextInt());
    }
}
