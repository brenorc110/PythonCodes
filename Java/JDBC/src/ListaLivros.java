import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
public class ListaLivros {
  public static void main(String[] args){
       //OBS.:PEGANDO A CONEXÃO POR AQUI,E NÂO PELA CLASSE !!!!!!!!!!!
       String stringDeConexao = "jdbc:mysql://localhost:3306/livraria"; 
       String usuario ="root";
       String senha = "ALUNOS";
       
       try{
           System.out.println("Abrindo conexão...");
           Connection conexao = 
                   DriverManager.getConnection(stringDeConexao, usuario, senha);
           String sql = "SELCT * FROM livro;";
           
           PreparedStatement comando = conexao.prepareStatement(sql);
           
           System.out.println("Executing comando...");
           ResultSet resultado = comando.executeQuery();
           
           System.out.println("Resultados encontrados: \n");
           while(resultado.next()){
               System.out.printf("%d : %s\n",
                       resultado.getInt("id"),
                       resultado.getString("titulo"),
                       resultado.getDouble("preco"),
                       resultado.getInt("id_ed"));
           }
           System.out.println("\nFechando conexao...");
           conexao.close();
       }catch (Exception e){
           e.printStackTrace();
       }
  }
}

