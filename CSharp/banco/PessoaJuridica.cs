using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace banco
{
    class PessoaJuridica : Pessoa
    {
        private string nome;

        public string GetNome()
        { return nome; }

        public void SetNome(string value)
        { nome = value; }
        private string razaoSocial;

        public string RazaoSocial
        {
            get { return razaoSocial; }
            set { razaoSocial = value; }
        }
        private string limite;

        public string Limite
        {
            get { return limite; }
            set { limite = value; }
        }
        private double Saldo;

        public double Saldo1
        {
            get { return Saldo; }
            set { Saldo = value; }
        }
        public double CalcularTarifa()
        {
            return Saldo * 0.2;


        }
        
}

    

        
}