
class TestaConta {

    public static void main(String[] args) {
        conta c1 = new conta();
        c1.saldo = 1000.0;
        c1.limite = 500.0;
        c1.numero = 1;

        //Chama o método deposita:
        c1.deposita(1000);
        c1.imprimeExtrato();
        c1.saca(100);
        double saldoDisponivel = c1.consultaSaldoDisponivel();
        System.out.println("Saldo disponível= " + saldoDisponivel);

        //conta c2 = new conta();
        //c2.saldo = 1000.0;
        //c2.limite = 500.0;
        //c2.numero = 1;

        //System.out.println(c1.saldo);
        //System.out.println(c1.limite);
        //System.out.println(c1.numero);

        //System.out.println(c2.saldo);
        //System.out.println(c2.limite);
        //System.out.println(c2.numero);


        conta origem = new conta();
        origem.saldo = 1000;
        conta destino = new conta();
        origem.transfere(destino, 500);
    }
}
