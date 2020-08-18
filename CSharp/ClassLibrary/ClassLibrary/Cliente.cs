using System;

namespace ClassLibrary
{
    public class Cliente
    {
        public void Andar(int x, int y)
        {
            Console.WriteLine("Walking in the rain...");
        }
        public void Mijar(){
            Console.WriteLine("Peeing...");
        }
        private string cpnj;

        public string Cpnj
        {
            get { return cpnj; }
            set { cpnj = value; }
        }
        private string descricao;

        public string Descricao
        {
            get { return descricao; }
            set { descricao = value; }
        }
        private string razaoSocial;

        public string RazaoSocial
        {
            get { return razaoSocial; }
            set { razaoSocial = value; }
        }
        enum TipoCliente { Fornecedor , Atacadista, ConsumidorDireto };
        
    }
}
