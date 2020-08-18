using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MySql.Data.MySqlClient;

namespace modeloDAO
{
    class PacienteDao
    {
        private MySqlConnection conn;

        public PacienteDao(MySqlConnection conn)
        {
            this.conn = conn;
        }

        public bool Gravar(PacienteModelo paciente)
        {
            MySqlCommand cmdBusca = new MySqlCommand("SELECT count(*) from tb_pacientes where idPaciente = '" + paciente.Id + "'", conn);

            int quantidade = Convert.ToInt32(cmdBusca.ExecuteScalar());

            MySqlCommand cmd;

            if (quantidade > 0)
            {
                cmd = new MySqlCommand("UPDATE tb_pacientes " +
                                                            " SET NomePaciente = @nome, EnderecoPaciente = @endereco, TelefonePaciente =  @telefone" +
                                                            " WHERE idPaciente = '" + paciente.Id + "'", conn);
            }
            else
            {
                cmd = new MySqlCommand("INSERT INTO tb_pacientes " +
                                                            " (NomePaciente, EnderecoPaciente, TelefonePaciente)" +
                                                            " VALUES (@nome, @endereco, @telefone)", conn);
            }

            MySqlParameter pNome = new MySqlParameter("nome", paciente.Nome);
            MySqlParameter pEndereco = new MySqlParameter("endereco", paciente.Endereco);
            MySqlParameter pTelefone = new MySqlParameter("telefone", paciente.Telefone);

            cmd.Parameters.Add(pNome);
            cmd.Parameters.Add(pEndereco);
            cmd.Parameters.Add(pTelefone);

            int celulasAfetadas = cmd.ExecuteNonQuery();

            if (celulasAfetadas > 0)
                return true;
            else
                return false;
        }

        public bool Apagar(PacienteModelo paciente)
        {
            return this.Apagar(paciente.Id);
        }

        public bool Apagar(int id)
        {
            MySqlCommand cmd = new MySqlCommand("DELETE FROM tb_pacientes " +
                                                " WHERE IdPaciente = '" + id + "'", conn);

            int celulasAfetadas = cmd.ExecuteNonQuery();

            if (celulasAfetadas > 0)
                return true;
            else
                return false;
        }

        public PacienteModelo Retornar(int id)
        {
            MySqlCommand cmdBusca = new MySqlCommand("SELECT * from tb_pacientes where idPaciente = '" + id + "'", conn);
            MySqlDataReader dr = cmdBusca.ExecuteReader();
            dr.Read();

            PacienteModelo paciente = new PacienteModelo();
            paciente.Id = Convert.ToInt32(dr["idPaciente"]);
            paciente.Nome = dr["NomePaciente"].ToString();
            paciente.Endereco = dr["EnderecoPaciente"].ToString();
            paciente.Telefone = dr["TelefonePaciente"].ToString();

            return paciente;
        }

        public List<PacienteModelo> RetornarTodos()
        {
            MySqlCommand cmdBusca = new MySqlCommand("SELECT * from tb_pacientes", conn);
            MySqlDataReader dr = cmdBusca.ExecuteReader();

            List<PacienteModelo> lista = new List<PacienteModelo>();

            while (dr.Read())
            {
                int id = Convert.ToInt32(dr["idPaciente"]);
                string nome = dr["NomePaciente"].ToString();
                string endereco = dr["EnderecoPaciente"].ToString();
                string telefone = dr["TelefonePaciente"].ToString();
                
                lista.Add(new PacienteModelo(id, nome, endereco, telefone));
            }

            return lista;
        }
    }
}
