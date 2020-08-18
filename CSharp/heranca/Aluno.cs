using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace heranca
{
    class Aluno : Pessoa 
    {
        private string serie;

        public string Serie
        {
            get { return serie; }
            set { serie = value; }
            //OBS:Set = Definir.
        }
        private string turma;

        public string Turma
        {
            get { return turma; }
            set { turma = value; }
        }

    }
}
