using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using MySql.Data.MySqlClient;

namespace Sistema
{
    public class BD
    {
        private MySqlConnection conexao;

        public BD()
        {
            conexao = new MySqlConnection("server=LocalHost;User Id=root;password=ALUNOS;" +
                                    "Persist Security Info=True;database=test");
        }

        public MySqlConnection Connect()
        {
            conexao.Open();

            return conexao;
        }

        public void Disconnect()
        {
            conexao.Close();
        }
    }
}