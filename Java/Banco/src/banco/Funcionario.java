
public class Funcionario {

    private int codigo;
    private String nome;
    private double salario;
    static double valeRefeicaoDiario;// O STATIC torna-o um atributo de classe,fixo,e não 
    //mais só da instância aumentaSalario():

    void aumentaSalario(double aumento) {
        this.setSalario(this.getSalario() + aumento);
    }

    /**
     * @return the nome
     */
    public String getNome() {
        return nome;
    }

    /**
     * @param nome the nome to set
     */
    public void setNome(String nome) {
        this.nome = nome;
    }

    /**
     * @return the salario
     */
    public double getSalario() {
        return salario;
    }

    /**
     * @param salario the salario to set
     */
    public void setSalario(double salario) {
        this.salario = salario;
    }

    /**
     * @return the codigo
     */
    public int getCodigo() {
        return codigo;
    }

    /**
     * @param codigo the codigo to set
     */
    public void setCodigo(int codigo) {
        this.codigo = codigo;
    }
}
