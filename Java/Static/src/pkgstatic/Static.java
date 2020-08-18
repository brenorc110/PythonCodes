package pkgstatic;

public class Static {

    Funcionarios f = new Funcionarios();

    public static void main(String[] args) {

        Funcionarios.vrd = 15;

    }

    static void reajustaVRD(double taxa) {
        System.out.println("VRD = " + Funcionarios.vrd);
        Funcionarios.vrd += Funcionarios.vrd * taxa;
        System.out.println("VRD REAJUSTADO = " + Funcionarios.vrd);
    }
}
