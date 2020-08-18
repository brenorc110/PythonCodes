using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace EngenSoftware
{
    public partial class Form2 : Form
    {
        public Form2()
        {
            InitializeComponent();
        }

        private void btnConverter_Click(object sender, EventArgs e)
        {
            switch (txtBox1.Text)
            {
                case "1":
                    lbl3.Text = "janeiro!";
                    break;
                case "2":
                    lbl3.Text = "fevereiro!";
                    break;
                case "3":
                    lbl3.Text = "março!";
                    break;
                case "4":
                    lbl3.Text = "abril!";
                    break;
                case "5":
                    lbl3.Text = "maio!";
                    break;
                case "6":
                    lbl3.Text = "junho!";
                    break;
                case "7":
                    lbl3.Text = "julho!";
                    break;
                case "8":
                    lbl3.Text = "agosto!";
                    break;
                case "9":
                    lbl3.Text = "setembro!";
                    break;
                case "10":
                    lbl3.Text = "outubro!";
                    break;
                case "11":
                    lbl3.Text = "novembro!";
                    break;
                case "12":
                    lbl3.Text = "dezembro!";
                    break;
                default:
                    lbl3.Text = " errou!";
                    break;
            }
             


        }

        private void btnSair_Click(object sender, EventArgs e)
        {
            this.Close();
        }
    }
}
