using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace Calculadora1
{
    public partial class Calculadora : Form
    {
        string valor_antigo = "";
        string operaçao = "";



        public Calculadora()
        {
            InitializeComponent();
        }

        private void Calculadora_Load(object sender, EventArgs e)
        {
            MessageBox.Show("Welcome 2 Calculadora!!!", "Parabéns!!! ", MessageBoxButtons.OK, MessageBoxIcon.Asterisk);
        }

        private void button1_click(object sender, EventArgs e)
        {
            screen.Text = screen.Text + "1";
        }

        private void button2_click(object sender, EventArgs e)
        {
            screen.Text = screen.Text + "2";
        }

        private void button3_click(object sender, EventArgs e)
        {
            screen.Text = screen.Text + "3";
        }

        private void button4_click(object sender, EventArgs e)
        {
            screen.Text = screen.Text + "4";
        }

        private void button5_click(object sender, EventArgs e)
        {
            screen.Text = screen.Text + "5";
        }

        private void button6_click(object sender, EventArgs e)
        {
            screen.Text = screen.Text + "6";
        }

        private void button7_click(object sender, EventArgs e)
        {
            screen.Text = screen.Text + "7";
        }

        private void button8_click(object sender, EventArgs e)
        {
            screen.Text = screen.Text + "8";
        }

        private void button9_click(object sender, EventArgs e)
        {
            screen.Text = screen.Text + "9";
        }

        private void button0_click(object sender, EventArgs e)
        {
            screen.Text = screen.Text + "0";
        }

        private void more_click(object sender, EventArgs e)
        {
            valor_antigo = screen.Text;
            operaçao = "+";
            screen.Text = "";
        }

        private void minus_click(object sender, EventArgs e)
        {
            valor_antigo = screen.Text;
            operaçao = "-";
            screen.Text = "";
        }

        private void mult_click(object sender, EventArgs e)
        {
            valor_antigo = screen.Text;
            operaçao = "*";
            screen.Text = "";
        }

        private void div_click(object sender, EventArgs e)
        {
            valor_antigo = screen.Text;
            operaçao = "/";
            screen.Text = "";
        }

        private void raiz_click(object sender, EventArgs e)
        {
            double numero_atual = Convert.ToDouble(screen.Text);
            double raiz = Math.Sqrt(numero_atual);

            screen.Text = raiz.ToString();
        }
		private void potencia_click(object sender, EventArgs e) {
			double numero_atual = Convert.ToDouble(screen.Text);
			double potencia = Math.Pow(2, numero_atual);

			screen.Text = potencia.ToString();
		}

        private void igual_click(object sender, EventArgs e)
        {
            int valor1 = Convert.ToInt32(valor_antigo);
            int valor2 = Convert.ToInt32(screen.Text);

            int valor_final = 0;


            if (operaçao == "+")
            {
                valor_final = valor1 + valor2;
            }
            else if (operaçao == "-")
            {
                valor_final = valor1 - valor2;
            }
            else if (operaçao == "*")
            {
                valor_final = valor1 * valor2;
            }
            else if(operaçao == "/")
            {
                valor_final = valor1 / valor2;
            }


            screen.Text = valor_final.ToString();

        }

        private void limpa_tela_click(object sender, EventArgs e) {
           screen.Text = "";
        }
        }
    }

