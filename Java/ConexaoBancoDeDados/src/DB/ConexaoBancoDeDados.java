package DB;



import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import Tela.Pessoa;
import Tela.rg;
import javax.swing.JOptionPane;

public class ConexaoBancoDeDados{

    private Connection conexao;
    private final String URLDB = "jdbc:mysql://localhost:3306/tx";
    private final String usuario = "root";
    private final String senha = "root";

    public ConexaoBancoDeDados() {
        try {
            Class.forName("com.mysql.jdbc.Driver");
            conexao = DriverManager.getConnection(URLDB, usuario, senha);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void inserir(Pessoa ps) {
      
        Statement inserir;
        PreparedStatement stmt = null;
        
        
        try {
            //inserir = conexao.createStatement();
            //Preencher os demis campos abaixo e suas interrogacoes
            stmt = conexao.prepareStatement("INSERT INTO tb_pessoas(nome,dt_data,cpf,rg,nt,n_mae,n_pai,patri, nacionalidade, data_em, orgao_em, pais_doc, fone_res) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)");
            stmt.setString(1,ps.getNome());
            stmt.setString(2,ps.getDataNasc());
            stmt.setDouble(3,ps.getCpf());
            stmt.setDouble(5,ps.getRg().getNumero());
            stmt.setString(5,ps.getNt());
            stmt.setString(6,ps.getNomeMae());
            stmt.setString(7,ps.getNomePai());
            stmt.setDouble(8,ps.getPatrimonio());
            stmt.setString(9, ps.getNacionalidade());
            stmt.setString(10, ps.getDataEmissao());
            stmt.setString(11, ps.getOrgaoEmissor());
            stmt.setString(12, ps.getUf());
            stmt.setString(13, ps.getPais_doc());
            stmt.setString(14, ps.getFone_res());
            
            stmt.executeUpdate();
            JOptionPane.showMessageDialog(null,"Dados inseridos com sucesso!");
            
        } catch (SQLException ex) {
            JOptionPane.showMessageDialog(null,"Erro no Insert no Banco!"+ex);
        }
    }

    public void update() {
        Statement comando;

        try {
            comando = conexao.createStatement();
            comando.execute("UPDATE estado SET estadoNome ='Rio de Janeiro' WHERE idEstado = 2");
        } catch (SQLException ex) {
            
        }
    }
    public void delete() {
        Statement comando;

        try {
            comando = conexao.createStatement();
            comando.execute("delete from estado WHERE idEstado = 3");
        } catch (SQLException ex) {
           
        }
    }
    public void selecionar() {
        try {
            Statement comando;
            comando = conexao.createStatement();
            ResultSet res = comando.executeQuery("SELECT * FROM estado");

            while (res.next()) {
                System.out.println(res.getString("estadoNome"));
            }
        } catch (SQLException ex) {
          
        }
    }
    
    public void prepareSelect(){
        try {
            PreparedStatement psSelect= this.conexao.prepareStatement("Select * from pessoas");
           // psSelect.setString(1, "1");
            ResultSet res = psSelect.executeQuery();
            
            while (res.next()) {
                System.out.println(res.getString("Nome"));
            }
            
        } catch (SQLException ex) {
            
        }
        
    }
    
    
 
    public static void main(String[] args) {
        ConexaoBancoDeDados conexao = new ConexaoBancoDeDados();
         conexao.prepareSelect();
    }

    
}