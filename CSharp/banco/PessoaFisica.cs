using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace banco
{
    class PessoaFisica : Pessoa
    {
        /*
        private string nome;

        public string Nome
        {
            get { return nome; }
            set { nome = value; }
        }
        */
        private double saldo;

        public double Saldo
        {
            get { return saldo; }
            set { saldo = value; }
        }
        public double CalcularTarifa()
        {
            return Saldo * 0.1;


        }
        
     }

}
