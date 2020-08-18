using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;
using MySql.Data.MySqlClient;


namespace WebApplication1
{
    public partial class Login : System.Web.UI.Page
    {
        protected void Page_Load(object sender, EventArgs e)
        {

        }

        protected void btLogin_Click(object sender, EventArgs e)
        {

            MySqlConnection cnn = new MySqlConnection("server=localhost;User Id=root;password=ALUNOS;" +
                                "Persist Security Info=True;database=prog");
           
            cnn.Open();

            MySqlCommand cmd = new MySqlCommand("SELECT  count(*) from usuarios WHERE email = '" + txEmail.Text + "' AND senha = '" + txSenha.Text + "' AND ativo = 1", cnn);
            
            object qt = cmd.ExecuteScalar();

            cnn.Close();

            int quantidade = Convert.ToInt32(qt);

         //if (txEmail.Text == "joao@exemplo.com" && txSenha.Text == "123")

            if ((int)quantidade > 0)
            {
                Session["logado"] = true;
                Response.Redirect("AcessoRestrito.aspx");
            }

            else
            {
                Session["logado"] = false;
                Response.Redirect("AcessoNegado.aspx");
            }

        }
    }
}