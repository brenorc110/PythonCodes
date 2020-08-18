using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

namespace N1
{
    public partial class WebFormCont : System.Web.UI.Page
    {


        protected void Page_Load(object sender, EventArgs e)
        {
            if (Session.Count > 0)
            {
                if (!Session["logado"].Equals(true))
                {
                    Response.Redirect("Erro.aspx");
                }
                else
                {
                    Response.Redirect("WebFormCont.aspx");
                }
            }
        }
    }
}