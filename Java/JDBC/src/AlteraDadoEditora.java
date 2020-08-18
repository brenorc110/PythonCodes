import java.sql.Connection;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.Scanner;
public class AlteraDadoEditora {
    public static void main(String[] args){
       
       Scanner entrada = new Scanner(System.in);
       
       try{
           //EX.1(Com o tratamento de Strings):
           Connection conexao = ConnectionFactory.createConnection();
               System.out.println("Abrindo conexão...");
               System.out.println("Digite o nome da editora:");
               String nome = entrada.nextLine();
               
               System.out.println("Digite o ID da editora");
               String id = entrada.nextLine();
               
               System.out.println("Digite o valor da editora a ser alterado: nome ou email");
               String alt = entrada.nextLine();
               if("nome".equals(alt)){
                   System.out.println("Digite o novo nome:");
                   String novonome = entrada.nextLine();
                   String sql = ("UPDATE editora " + " SET nome = '" + novonome + "'" +
                                                            " WHERE id_ed = '" + id + "'");
                   
                   //Cria um comando a partir do código SQL:
                       PreparedStatement comando = conexao.prepareStatement(sql);
                       
                        System.out.println("Executing comando...");
                        ResultSet resultado;
                        resultado = comando.executeQuery();
               } else {
                   System.out.println("Digite o novo e-mail:");
                   String novoem = entrada.nextLine();
                   String sql = ("UPDATE editora " + " SET email = '" + novoem + "'" +
                                                            " WHERE id_ed = '" + id + "'");
                   
                   //Cria um comando a partir do código SQL:
                       PreparedStatement comando = conexao.prepareStatement(sql);
                       
                   System.out.println("Executing comando...");
                   ResultSet resultado;
                   resultado = comando.executeQuery();
               }
          }catch (Exception e){ 
          e.printStackTrace();
        
    }
    }
    
}
