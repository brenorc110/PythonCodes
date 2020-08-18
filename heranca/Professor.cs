using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace heranca
{
    class Professor : Pessoa
    {
        private string a_atuacao;

        public string AreaAtuacao
        {
            get { return a_atuacao; }
            set { a_atuacao = value; }
        }
        private double c_horaria;

        public double CargaHoraria
        {
            get { return c_horaria; }
            set { c_horaria = value; }
        }
        public void registrarDiario()
        {


        }

    }
}
