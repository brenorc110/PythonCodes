import java.util.concurrent.ConcurrentHashMap;
public class Pilha {
    private ConcurrentHashMap pilha;
    private int controle;
    public Pilha(){
        pilha = new ConcurrentHashMap();
        controle=-1;        
    }
    public void empilha(String dado){        
        pilha.put(++controle, dado);    
    }
    public String desempilha(){                        
        String dado="";
        if (!vazio()){            
            dado = topoGet();
            pilha.remove(controle--);                        
        }
        return dado;
    }
    public String topoGet(){                        
        String dado="";
        if (!vazio()){
            dado= pilha.get(controle).toString();                        
        }
        return dado;
    }
    public boolean vazio(){
        return pilha.isEmpty();
    }
}
