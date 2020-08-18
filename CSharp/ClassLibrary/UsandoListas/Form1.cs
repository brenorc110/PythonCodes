using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using ClassLibrary.Cliente;

namespace UsandoListas
{
    public  partial class Form1 : Form
    {
        public Form1 ()
        {
            InitializeComponent();
        }

        
            

        private void button2_Onclick(object sender, EventArgs e)
        {
            dataGridView1.DataSource = listaDeClientes;
            dataGridView1.Refresh();
        }

        private void button1_OnClick(object sender, EventArgs e)
        {
            string t1 = boxCpnj.Text.ToString();
            string t2 = boxDescricao.Text.ToString();
            string t3 = boxRazaoSocial.Text.ToString();
            dataGridView1.Text = t1 + t2 + t3;
        }

        
        
       
            
    }
}
