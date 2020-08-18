/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 * USANDO CONSTUTOR, THIS E ATRIBUTOS, INSTÃNCIAS
 * @author fsj.5947373464
 */
public class Aplicacao {
    public static void main(String[] args) {
        Pessoa pessoa1 = new Pessoa("Breno",444444444);
        pessoa1.cpf=00000000001;
        pessoa1.titulo=000000010;
        
        pessoa1.mae = new Pessoa("Sandra",555555555);
        System.out.println(pessoa1.mae.nome);
        pessoa1.falar();
        pessoa1.mae.falar();
        
        Pessoa pessoa2 = new Pessoa("Beto",666666666);
        pessoa2.cpf=00000000002;
        pessoa2.titulo=000000020;
        
    }
}
