using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace polimorfismo
{
    class CDB : IContaInvestimento
    {
        double saldo;

        public CDB()
        {
        }

        public double Saldo
        {
            get { return saldo; }
            set { saldo = value; }
        }

        public double Rendimento1 { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        public double CalcularCBD (double Saldo)
        {
            return Saldo;
        }

        public void Depositar(double valor)
        {
            throw new NotImplementedException();
        }
    }
}
