using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

namespace Sistema
{
    public partial class Painel : System.Web.UI.MasterPage
    {
        protected void Page_Load(object sender, EventArgs e)
        {
            if (!Session["conectado"].Equals(true))
            {
                Response.Redirect("/Login.aspx");
            }
        }
    }
}