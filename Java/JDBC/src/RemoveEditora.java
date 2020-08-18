import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.Scanner;
public class RemoveEditora {
    public static void main(String[] args){
       String stringDeConexao = "jdbc:mysql://localhost:3306/livraria"; 
       String usuario ="root";
       String senha = "ALUNOS";
       
       Scanner entrada = new Scanner(System.in);
       
       try{
           //EX.1(Com o tratamento de Strings):
           Connection conexao = DriverManager.getConnection(stringDeConexao, usuario, senha) ;
               System.out.println("Abrindo conexão...");
               System.out.println("Digite o nome da editora:");
               String nome = entrada.nextLine();
               
               System.out.println("Digite o ID da editora");
               String id = entrada.nextLine();
               String sql = ("DELETE FROM Editora "
                       + " WHERE id_ed = '" + id + "'") ;
                       
                       //Cria um comando a partir do código SQL:
                       PreparedStatement comando = conexao.prepareStatement(sql);
                       
                       System.out.println("Executing comando...");
           ResultSet resultado = comando.executeQuery();
           
       }catch (Exception e){ 
          e.printStackTrace();
        
    }
    }
    
}
