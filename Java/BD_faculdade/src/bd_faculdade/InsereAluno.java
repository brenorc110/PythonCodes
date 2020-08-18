/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package bd_faculdade;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.util.Scanner;

/**
 *
 * @author fsj.5947373464
 */
public class InsereAluno {
    public static void main(String[] args){
       Scanner entrada = new Scanner(System.in);
       
       try{
           //EX.1(Com o tratamento de Strings):
          //OBS.:PEGANDO A CONEXÃO PELA CLASSE,NÃO POR AQUI.
               System.out.println("Abrindo conexão...");
               Connection conexao = BD_faculdade.createConnection();
               System.out.println("Digite o nome do aluno:");
               String nome = entrada.nextLine();
               
               System.out.println("Digite o RA do aluno: ");
               String RA = entrada.nextLine();
               
               System.out.println("Digite o telefone do aluno: ");
               String telefone = entrada.nextLine();
               
               String sql = "INSERT INTO aluno (nome, RA, telefone) VALUES (?,?,?)";
                       
                       //Cria um comando a partir do código SQL:
                       PreparedStatement comando = conexao.prepareStatement(sql);
                       
                       //Add as entradas do usuário ao comando - As Strings são tratadas aqui(Sanitize):
                       comando.setString(1, nome);
                       comando.setString(1, RA);
                       comando.setString(1, telefone);
                       System.out.println("Executing comando...");
                       comando.execute();
                       
                       System.out.println("Executing comando...");
           
       }catch (Exception e){ 
          e.printStackTrace();
        
    }
    }
    
}