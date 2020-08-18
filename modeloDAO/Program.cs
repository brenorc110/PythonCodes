using System.Data.SqlClient;
using MySql.Data.MySqlClient;
using System;

namespace modeloDAO
{
    class Program
    {
        static void Main(string[] args)
        {
            MySqlConnection conn = 
                new MySqlConnection("server=LocalHost;User Id=root;password=ALUNOS;" +
                                    "Persist Security Info=True;database=db_consultorio");

            conn.Open();
            PacienteDao dao = new PacienteDao(conn);
            
            // Grava um novo paciente
            PacienteModelo p1 = new PacienteModelo();
            p1.Nome = "Jose";
            p1.Endereco = "Rua Angelo Bravini, 280";
            p1.Telefone = "1232093467";
            dao.Gravar(p1);

            // Atualiza informacoes do paciente
            PacienteModelo p2 = new PacienteModelo();
            p2.Id = 1;
            p2.Nome = "Jose Maria";
            p2.Endereco = "Rua Papa Joao XIII";
            p2.Telefone = "1232093467";
            dao.Gravar(p2);

            // Apaga um paciente            
            // dao.Apagar(1);
            
            // Retorna um paciente especifico
            // PacienteModelo p = dao.Retornar(5);

            // Retorna varios paciente
            foreach (PacienteModelo p in dao.RetornarTodos())
            {
                Console.WriteLine(p);
            }

            conn.Close();

            Console.ReadKey();
        }
    }

    internal class PacienteModelo
    {
        public int Id;
        public string Nome;
        public string Endereco;
        public string Telefone;

        public PacienteModelo(int id, string nome, string endereco, string telefone)
        {
            Id = id;
            Nome = nome;
            Endereco = endereco;
            Telefone = telefone;
        }
    }
}
