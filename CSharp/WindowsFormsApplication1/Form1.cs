using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace WindowsFormsApplication1
{
    public  partial class Form1 : Form
    {
        public Form1 ()
        {
            InitializeComponent();
        }
        
        private void Form1_Load(object sender, EventArgs e)
        {
            lblEntidade.Text = "";
            textBox1.Enabled = false;
            
        }

        private void chkMedico_CheckedChanged(object sender, EventArgs e)
        {
            
        }

        private void rdMedico_CheckedChanged(object sender, EventArgs e)
        {
            lblEntidade.Text = "CRM: ";
            textBox1.Enabled = true;
        }

        private void rdTecnico_CheckedChanged(object sender, EventArgs e)
        {
            lblEntidade.Text = "CREA: ";
            textBox1.Enabled = true;
        }

        private void radioButton1_CheckedChanged(object sender, EventArgs e)
        {
            lblEntidade.Text = "";
            textBox1.Enabled = false;
        }

        private void radioButton2_CheckedChanged(object sender, EventArgs e)
        {
            lblEntidade.Text = "RA: ";
            textBox1.Enabled = true;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            // Cliquei no botão Mostrar Dados...
            // E AGORA ?


            // Vamos instanciar classes ESPECIALISTAS (médico, tecnico, etc) em uma 
            // mais GENERICA (Pessoa).

            // Então, podemos fazer o seguinte....

            Pessoa p = null;

            //Ainda não sabemos qual o tipo da PESSOA !!!!!! Agora vamos instanciar, 
            //dado a seleção....

            if (rdMedico.Checked == true)
            {
                //Médico selecionado...
                p = new Medico();
            }
            else if (rdTecnico.Checked == true)
            {
                //Tecnico Selecionado...
                p = new Tecnico();
            }
            else if (radioButton1.Checked == true)
            {
                //Pessoa Comum Selecionada...
                p = new PessoaComum();
            }
            else if (radioButton2.Checked == true)
            {
                //Aluno selecionado
                p = new Aluno();
            }
            else
            {
                MessageBox.Show("Selecionar pelo menos um tipo de pessoa!");
            }


            p.Nome = "Edizon";
            p.Idade = 32;
            p.Endereco = "Rua das Acacias, 75";
            p.Sexo = "Masculino";
            

            MessageBox.Show("Tipo selecionado: " + p.GetType().ToString() +
                            Environment.NewLine + 
                            "Valores: " + p.MostrarValores());
        }
        
        
        
       
            
    }
}
