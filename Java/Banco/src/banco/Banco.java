public class Banco {

    Gerente g = new Gerente();
    Funcionario f = new Funcionario();
    Funcionario f1 = g;

    public static void main(String[] args) {

        Funcionario.valeRefeicaoDiario = 15;
    }

    static void reajustaValeRefeicaoDiario(double taxa) {
        Funcionario.valeRefeicaoDiario += Funcionario.valeRefeicaoDiario * taxa;
    }
    Emprestimo emp;
    seguroVeicular sv;

    public Banco() {
        this.sv = new seguroVeicular();
        this.emp = new Emprestimo();
        System.out.println("Emprestimo: " + emp.calculaTaxa());
        System.out.println("" + sv.calculaTaxa());
    }
}
