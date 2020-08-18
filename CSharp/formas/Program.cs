using System;
/*
Passos para rodar:
1- Especificar a localizacao do .csproj em tasks.json item publish.
2- Especificar a localizacao da dll gerada pelo item anterior.
3- Run without debugging.
*/

namespace formas
{
    class Program
    {
        static void Main(string[] args)
        {
            Quadrado q = new Quadrado();
            Console.WriteLine(" Insira o valor dos lados do qudrado: ");
            q.lado = Convert.ToDouble(Console.ReadLine());
            Console.WriteLine("Calculo de area: " + q.calcularArea(q.lado));
            Console.WriteLine("Calculo de perimetro: " + q.calcularPerimetro(q.lado));
        }
    }
}

