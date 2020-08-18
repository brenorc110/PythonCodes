public class Pessoa {
    public String nome;
    public long rg;
    public long cpf;
    public long titulo;
    
    public Pessoa mae;
    
    Pessoa(String nome, long rg){
        this.nome = nome;
        this.rg = rg;
    }
    
    public void falar(){
       System.out.println(this.nome + " falando.");
    }
}

/*
public class Pessoa {
public int idade;
public String nome;
Pessoa(){
	System.out.println("Eu sou o construtor, serei chamado primeiro!");
}
public void fazerAniversario(){
this.idade++;
//THIS faz referência a atributo da classe, nunca uma variável deste escopo.
//this = deste. 
}

public static void main(String []args){
Pessoa p = new Pessoa();
p.nome="Breno";
p.idade=22;
p.fazerAniversario();
System.out.println(p.idade);
}
}*/