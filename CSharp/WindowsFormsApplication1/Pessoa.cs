using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace WindowsFormsApplication1
{
    public class Aluno : Pessoa
    {
        public int RA { get; set; }

        public string MostrarValores()
        {
            return base.MostrarValores() + " / RA -> " + RA;
        } 
    }
    public class Medico : Pessoa
    {
        public int CRM { get; set; }

        public string MostrarValores()
        {
            return base.MostrarValores() + " / CRM -> " + CRM;
        } 
    }
    public class Tecnico : Pessoa
    {
        public int CREA { get; set; }

        public string MostrarValores()
        {
            return base.MostrarValores() + " / CREA -> " + CREA;
        } 
    }
    public class PessoaComum : Pessoa
    {
        public string TempoDisponivel { get; set; }
    }
    public abstract class Pessoa
    {
        //Atributos
        string _nome;
        int _idade;
        public string Sexo { get; set; } 
        public string Endereco { get; set; }

        public string Nome
        {
            get { return _nome; }
            set { _nome = value; }
        }
        public int Idade
        {
            get { return _idade; }
            set {
                if (value < 0)
                    _idade = 0;
                else 
                    _idade = value; 
            }
        }

        public void FazerAniversario()
        {
            _idade++;
        }

        public string MostrarValores()
        {
            return "Nome ->     "   + this.Nome +
                " / Endereco -> "   + this.Endereco +
                " / Idade    -> "   + this.Idade +
                " / Sexo     -> "   + this.Sexo;
        }
     

    }
}
