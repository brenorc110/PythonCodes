using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

public partial class _Default : System.Web.UI.Page
{
    protected void Page_Load(object sender, EventArgs e)
    {

    }
    protected void btnGravar_Click(object sender, EventArgs e)
    {

        if (!string.IsNullOrEmpty(txtNome.Text.Trim()) && !string.IsNullOrEmpty(txtTitulo.Text.Trim()))
        {

            Response.Cookies["Livro"]["Autor"] = txtNome.Text;
            Response.Cookies["Livro"]["Titulo"] = txtTitulo.Text;
            Response.Cookies["Livro"].Expires = DateTime.Now.AddDays(2); lblMensagem.Text = string.Empty;

        }

        else
        {

            lblMensagem.Text = "Preencha todos os campos do formulário."; txtNome.Focus();

        }

    }
    protected void btnLer_Click(object sender, EventArgs e)
    {
        if (Request.Cookies["Livro"] != null)
        {
            if (Request.Cookies["Livro"]["Autor"] != null &&
                Request.Cookies["Livro"]["Titulo"] != null)
            {
                lblSaidaAutor.Text = "Autor: " +
                Server.HtmlEncode(Request.Cookies["Livro"]["Autor"]);
                lblSaidaTitulo.Text = "Livro: " +
               Server.HtmlEncode(Request.Cookies["Livro"]["Titulo"]);
            }
        }
    }
    protected void btnMatar_Click(object sender, EventArgs e)
    {
        Response.Cookies["Livro"].Expires = DateTime.Now.AddDays(-1d);
    }
}





