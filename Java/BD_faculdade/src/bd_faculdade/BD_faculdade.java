/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package bd_faculdade;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

/**
 *
 * @author fsj.5947373464
 */
public class BD_faculdade {
public static Connection createConnection(){
      String stringDeConexao = "jdbc:mysql://localhost:3306/faculdade";
      String usuario = "root";
      String senha = "";
      
      Connection conexao;
      conexao = null;
      try{
          conexao = DriverManager.getConnection(stringDeConexao, "root", "ALUNOS");
          return conexao;
      }catch(SQLException e){
          e.printStackTrace();
      }
     return conexao; 
 }
    
}
