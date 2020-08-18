using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;
using MySql.Data.MySqlClient;

namespace Sistema.Sistema
{
    public partial class UsuariosEditar : System.Web.UI.Page
    {
        private MySqlConnection conexao;
        private int id_corrente = 0;

        protected void Page_Load(object sender, EventArgs e)
        {
            BD bd = new BD();

            conexao = bd.Connect();

            CarregarRegistro();
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
                
                if (!Page.IsPostBack)
                {
                    txNome.Text = dr["nome"].ToString();
                    txEmail.Text = dr["email"].ToString();
                    txSenha.Text = dr["senha"].ToString();
                    ckAtivo.Checked = Convert.ToBoolean(dr["ativo"]);
                }

                dr.Close();
            }
        }
        
        protected void btSalvar_Click(object sender, EventArgs e)
        {
            if (id_corrente > 0)
            {
                MySqlCommand cmd = new MySqlCommand("UPDATE tb_usuarios SET nome = @nome, email = @email, senha = @senha, ativo = @ativo WHERE id = @id", conexao);

                MySqlParameter pNome = new MySqlParameter("nome", txNome.Text);
                MySqlParameter pEmail = new MySqlParameter("email", txEmail.Text);
                MySqlParameter pSenha = new MySqlParameter("senha", txSenha.Text);
                MySqlParameter pAtivo = new MySqlParameter("ativo", ckAtivo.Checked);
                MySqlParameter pId = new MySqlParameter("id", id_corrente);

                cmd.Parameters.Add(pNome);
                cmd.Parameters.Add(pEmail);
                cmd.Parameters.Add(pSenha);
                cmd.Parameters.Add(pAtivo);
                cmd.Parameters.Add(pId);

                int quant = (int)cmd.ExecuteNonQuery();
            }

            Response.Redirect("/Sistema/Usuarios.aspx");    
        }

        protected void btnCancelar_Click(object sender, EventArgs e)
        {
            Response.Redirect("/Sistema/Usuarios.aspx");
        }
    }
}