using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;
using MySql.Data.MySqlClient;

namespace Sistema.Sistema
{
    public partial class UsuariosExcluir : System.Web.UI.Page
    {
        private MySqlConnection conexao;
        private int id_corrente = 0;

        protected void Page_Load(object sender, EventArgs e)
        {
            BD bd = new BD();

            conexao = bd.Connect();

            CarregarRegistro();
            ExcluirRegistro();
        }

        private void CarregarRegistro()
        {
            string id = Request.QueryString["id"];

            if (id != null)
            {
                MySqlCommand cmd = new MySqlCommand("SELECT * FROM tb_usuarios WHERE id = @id", conexao);
                MySqlParameter pId = new MySqlParameter("id", id);
                cmd.Parameters.Add(pId);

                MySqlDataReader dr = cmd.ExecuteReader();
                dr.Read();

                id_corrente = Convert.ToInt32(dr["id"]);
                
                dr.Close();
            }
        }
        
        private void ExcluirRegistro()
        {
            if (id_corrente > 0)
            {
                MySqlCommand cmd = new MySqlCommand("DELETE FROM tb_usuarios WHERE id = @id", conexao);
                
                MySqlParameter pId = new MySqlParameter("id", id_corrente);
                cmd.Parameters.Add(pId);

                int quant = (int)cmd.ExecuteNonQuery();

                Response.Redirect("/Sistema/Usuarios.aspx");    

            }
        }
    }
}