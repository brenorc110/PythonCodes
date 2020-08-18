import java.sql.Connection;

import java.sql.PreparedStatement;
import java.util.Scanner;
public class InsereEditora {
    public static void main(String[] args){
       Scanner entrada = new Scanner(System.in);
       
       try{
           //EX.1(Com o tratamento de Strings):
          //OBS.:PEGANDO A CONEXÃO PELA CLASSE,NÃO POR AQUI.
               System.out.println("Abrindo conexão...");
               Connection conexao = ConnectionFactory.createConnection();
               System.out.println("Digite o nome da editora:");
               String nome = entrada.nextLine();
               
               System.out.println("Digite o e-mail da editora");
               String email = entrada.nextLine();
               String sql = "INSERT INTO Editora (nome, email) VALUES (?,?)";
                       
                       //Cria um comando a partir do código SQL:
                       PreparedStatement comando = conexao.prepareStatement(sql);
                       
                       //Add as entradas do usuário ao comando - As Strings são tratadas aqui(Sanitize):
                       comando.setString(1, nome);
                       comando.setString(1, email);
                       System.out.println("Executing comando...");
                       comando.execute();
                       
                       System.out.println("Executing comando...");
           
       }catch (Exception e){ 
          e.printStackTrace();
        
    }
    }
    
}
