
import java.text.SimpleDateFormat;
import java.util.Date;

/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author Casa
 */
public class ControleDePonto {

    public void registraEntrada(Funcionario f) {
        SimpleDateFormat sdf = new SimpleDateFormat("dd/MM/yyyy HH:mm:ss");
        Date agora = new Date();

        System.out.println("ENTRADA: " + f.getCodigo());
        System.out.println("DATA: " + sdf.format(agora));

    }

    public void registraSaida(Funcionario f) {
        SimpleDateFormat sdf = new SimpleDateFormat("dd/MM/yyyy HH:mm:ss");
        Date agora = new Date();

        System.out.println("SAÍDA: " + f.getCodigo());
        System.out.println("DATA: " + sdf.format(agora));

    }
}
