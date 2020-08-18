using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;
using MySql.Data.MySqlClient;

namespace Sistema.Sistema
{
    public partial class UsuariosNovo : System.Web.UI.Page
    {
        private MySqlConnection conexao;

        protected void Page_Load(object sender, EventArgs e)
        {
            BD bd = new BD();

            conexao = bd.Connect();
        }

        protected void btSalvar_Click(object sender, EventArgs e)
        {
            
            MySqlCommand cmd = new MySqlCommand("INSERT INTO tb_usuarios (nome, email, senha, ativo) VALUES (@nome, @email, @senha, @ativo)", conexao);

            MySqlParameter pNome = new MySqlParameter("nome", txNome.Text);
            MySqlParameter pEmail = new MySqlParameter("email", txEmail.Text);
            MySqlParameter pSenha = new MySqlParameter("senha", txSenha.Text);
            MySqlParameter pAtivo = new MySqlParameter("ativo", ckAtivo.Checked);

            cmd.Parameters.Add(pNome);
            cmd.Parameters.Add(pEmail);
            cmd.Parameters.Add(pSenha);
            cmd.Parameters.Add(pAtivo);

            int quant = (int)cmd.ExecuteNonQuery();
            

            Response.Redirect("/Sistema/Usuarios.aspx");    
        }

        protected void btnCancelar_Click(object sender, EventArgs e)
        {
            Response.Redirect("/Sistema/Usuarios.aspx");
        }
    }
}