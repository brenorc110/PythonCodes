using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace polimorfismo
{
    class Poupanca : IContaInvestimento
    {
        double saldo;

        public double Saldo
        {
            get { return saldo; }
            set { saldo = value; }
        }

        public double Rendimento1 { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        public double CalcularPoupança(double Saldo)
        {
            
            return Saldo;
        }

        public void Depositar(double valor)
        {
            throw new NotImplementedException();
        }
    }
}

