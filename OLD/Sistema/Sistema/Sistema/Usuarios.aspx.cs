using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;
using MySql.Data.MySqlClient;
using System.Data;

namespace Sistema.Sistema
{
    public partial class Usuarios : System.Web.UI.Page
    {
        private MySqlConnection conexao; 

        protected void Page_Load(object sender, EventArgs e)
        {
            BD bd = new BD();

            conexao = bd.Connect();
            
            AtualizarGrid();
        }

        private void AtualizarGrid()
        {
            MySqlCommand cmd = new MySqlCommand("SELECT * FROM tb_usuarios", conexao);

            MySqlDataAdapter da = new MySqlDataAdapter(cmd);

            DataTable usuarios = new DataTable();
            da.Fill(usuarios);

            gvRegistros.AutoGenerateColumns = false;
            gvRegistros.DataSource = usuarios;
            gvRegistros.DataBind();
        }
    }
}