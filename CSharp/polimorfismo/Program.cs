using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using polimorfismo;

namespace polimorfismo
{
    class Program
    {
        static void Main(string[] args)
        {
            IContaInvestimento ici = new Poupanca();
            Console.WriteLine("Insira a opção:1 Deposito,2 Saque");
            string g1 = Console.ReadLine();
            Int32 op = Convert.ToInt32(g1);
            Console.WriteLine("Insira o dia de hoje");
            string g2 = Console.ReadLine();
            Int32 dia = Convert.ToInt32(g2);
            Console.WriteLine("Insira o mês corrente");
            string g3 = Console.ReadLine();
            Int32 mes = Convert.ToInt32(g3);

            if (op == 1)
            {
                Console.WriteLine("Inf.o valor do depósito");
                string g4 = Console.ReadLine();
                double valor1 = Convert.ToDouble(g4);
                Console.WriteLine("Inf.o valor do Saldo");
                string g5 = Console.ReadLine();
                double valor2 = Convert.ToDouble(g5);
                double saldo = valor2 + valor1;
                Console.WriteLine("Seu saldo é : R$ " + saldo);
            }
            if (op == 2)
            {
                Console.WriteLine("Inf.o valor do depósito");
                string g4 = Console.ReadLine();
                double valor1 = Convert.ToDouble(g4);
                Console.WriteLine("Inf.o valor do Saldo");
                string g5 = Console.ReadLine();
                double valor2 = Convert.ToDouble(g5);
                double Saldo = valor2 - ((valor2*(0.01)*dia));
                Console.WriteLine("Seu saldo é : R$ " + Saldo);
            }   
        }
    }
}