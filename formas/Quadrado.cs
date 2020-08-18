using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace formas
{
    class Quadrado
    {
        public double lado;

        public double calcularArea(double lado) => lado * lado;
        public double calcularPerimetro(double lado) => 4 * lado;//ou double resultado = 4*lado;//return resultado ; 

        public override bool Equals(object obj)
        {
            return obj is Quadrado quadrado &&
                   lado == quadrado.lado;
        }
    }
}
