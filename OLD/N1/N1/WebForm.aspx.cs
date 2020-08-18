using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

namespace N1
{
    public partial class WebForm : System.Web.UI.Page
    {
        protected void Page_Load(object sender, EventArgs e)
        {

        }

        protected void btLogin_OnClick(object sender, EventArgs e)
        {
            if (tbEmail.Text == "joao@exemplo.com" &&
                tbSenha.Text == "123")
            {
                Session["logado"] = true;
                Response.Redirect("WebFormCont.aspx");
            }
            else
            {
                Session["logado"] = false;
                Response.Redirect("Erro.aspx");
            }

        }
    }
}