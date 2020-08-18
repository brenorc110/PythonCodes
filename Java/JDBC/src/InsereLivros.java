
import java.io.Console;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.util.Scanner;

public class InsereLivros {

    public static void main(String[] args) {
        //OBS.:PEGANDO A CONEXÃO POR AQUI,E NÂO PELA CLASSE !!!!!!!!!!!
        String stringDeConexao = "jdbc:mysql://localhost:3306/livraria";
        String usuario = "root";
        String senha = "ALUNOS";

        Scanner entrada = new Scanner(System.in);

        try {
            //EX.2(Método inseguro,sem tratamento das Strings e que cairá no SQL Injection):
            Connection conexao = DriverManager.getConnection(stringDeConexao, usuario, senha);
            System.out.println("Abrindo conexão...");
            System.out.println("Digite o título do livro:");
            String titulo = entrada.nextLine();

            //Lê as entradas do usuário:
            System.out.println("Digite o preco do livro");
            String pegapreco = entrada.nextLine();

            //Código SQL com marcadores para as entradas do usuário:
            String sql = "INSERT INTO Livro (titulo, preco)"
                    + "VALUES ('" + titulo + "','" + pegapreco + "')";

            //Cria um comando a partir do código SQL:
            PreparedStatement comando = conexao.prepareStatement(sql);

            System.out.println("Executing comando...");
            comando.execute();

            System.out.println("Executing comando...");

        } catch (Exception e) {
            e.printStackTrace();

        }
    }
}
