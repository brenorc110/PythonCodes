using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace polimorfismo
{
    public interface IContaInvestimento
    {
                double Saldo
        {
            get;
        }

        double Rendimento1
        {
            get { return Rendimento1; }
            set { Rendimento1 = value; }
        } 
        /*
        double TaxaAdministrativa
        {
            get { return taxaAdministrativa; }
            set { taxaAdministrativa = value; }
        }
        */


        double CalcularJuros(double valor)
        {
            double juros = (valor * (0.01));
            return juros;
        }

        void Depositar(double valor);
        double Sacar(double valor)
        {
            return valor;
        }

    }
}