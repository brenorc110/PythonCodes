using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace heranca
{
    class Program
    {
        static void Main(string[] args)
        {
            Pessoa pess = new Pessoa();
            Professor prof = new Professor();
            Aluno a = new Aluno();
            prof.Sx = "masculino";
            a.Turma =  "9A";
        }
    }
}
