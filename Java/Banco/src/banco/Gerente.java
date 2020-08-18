
class Gerente extends Funcionario {

    String nome;
    double salario;
    private int codigo;
    String user;
    String senha;

    void aumentaSalario() {
        this.aumentaSalario(0.1);
    }

    @Override
    void aumentaSalario(double taxa) {
        this.salario += this.salario * taxa;
    }

    /**
     * @return the codigo
     */
    @Override
    public int getCodigo() {
        return codigo;
    }

    /**
     * @param codigo the codigo to set
     */
    @Override
    public void setCodigo(int codigo) {
        this.codigo = codigo;
    }
}
