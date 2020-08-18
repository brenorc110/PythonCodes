using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace banco
{
    class Pessoa
    {
        private string nome;

        public string Nome
        {
            get { return nome; }
            set { nome = value; }
        }
        private string cpf;

        public string Cpf
        {
            get { return cpf; }
            set { cpf = value; }
        }
        private string sx;

        public string Sx
        {
            get { return sx; }
            set { sx = value; }
        }

        public double Saldo { get; private set; }

        public double CalcularTarifa(){
            return Saldo;
        }
    }
}
