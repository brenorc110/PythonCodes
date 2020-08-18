using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;
using MySql.Data.MySqlClient;

namespace Sistema
{
    public partial class Login1 : System.Web.UI.Page
    {
        private MySqlConnection conexao;

        protected void Page_Load(object sender, EventArgs e)
        {

        }

        protected void btLogin_Click(object sender, EventArgs e)
        {
            BD bd = new BD();

            conexao = bd.Connect();

            MySqlCommand cmd = new MySqlCommand("SELECT COUNT(*) FROM tb_usuarios WHERE email = @email AND senha = @senha AND ativo = 1", conexao);
            
            MySqlParameter pEmail = new MySqlParameter("email", txEmail.Text);
            MySqlParameter pSenha = new MySqlParameter("senha", txSenha.Text);
            cmd.Parameters.Add(pEmail);
            cmd.Parameters.Add(pSenha);

            int count = Convert.ToInt32(cmd.ExecuteScalar());

            bd.Disconnect();

            if (count > 0)
            {
                Session["conectado"] = true;
                Response.Redirect("/Sistema/Dashboard.aspx");
            }
            else
            {
                Session["conectado"] = false;
                Response.Redirect("/LoginError.aspx");
            }
        }
    }
}