public class HashLinear {

    private Hash[] tab;
    private int TAM_MAX;
  
    public HashLinear(int tam) {
      tab = new Hash[tam];
      TAM_MAX = tam;
      for(int i=0; i<tam; i++)   // inicializando
          tab[i] = new Hash(false);
    }
    
    private int funcaohash(String chave) {
          int v = (int) chave;
      return ( Math.abs(v) % TAM_MAX );
    }
    
    public void insere(String item) {
      
      if (cheia()) { // Se tabela cheia, imprime aviso e sai da função
         System.out.println("\n->ATENCAO Tabela cheia");
         return; // saida imediata da função, nao executa os comandos abaixo
      }
      
      int pos = funcaohash(item); // CALCULA POSIÇAO
      
      // INICIO ROTINA TRATAMENTO DE COLISAO
      if (tab[pos].ocupado!=true) {      } else {
          // se ocorreu colisao
          
          if (item.equals(tab[pos].item)) { // se a chave ja existe
              System.out.println("\n->ATENCAO Esse item ja foi cadastrado");
              return; // saida imediata da função
          }
          
          System.out.println("-> Ocorreu uma colisao na posicao " + pos);
          while (pos < TAM_MAX) {
              if (pos == TAM_MAX-1 ) pos = -1; // volta para o inicio da tabela
              pos++; // incremento mais um no indice
              if ( !tab[pos].ocupado ) // se a posição estiver livre
                  break; // sai do loop com o indice na posicao correta (pos sem colisão)
          } }
      // FIM ROTINA TRATAMENTO DE COLISAO
      
      tab[pos].item=item;
      tab[pos].ocupado = true;
      System.out.print("-> Inserido HASH[" + pos + "]");
    }
  
    public int busca(String chave) { // Recuperando um elemento
       int pos = funcaohash(chave);
       if (tab[pos].ocupado) {
          if (tab[pos].item.equals(chave)) // Se o campo esta ocupado e o nome e chave são iguais
         return pos; // saida imediata da função
  
             // INICIO ROTINA TRATAMENTO DE COLISAO (se o item e chave nao sao iguais)
             int iniciobusca = pos;
         while (pos < TAM_MAX) { // percorre proximas posições do vetor
               if (pos == TAM_MAX-1) pos=-1; // volta para o inicio da tabela
               pos++; // incremento mais um no indice
           if (tab[pos].ocupado && tab[pos].item.equals(chave)) // se o campo esta ocupado e o item foi encontrado
           return pos; // saida imediata da função
          if (pos == iniciobusca) return -1; // se percorreu tudo e nao encontrou
             }
        //FIM ROTINA TRATAMENTO DE COLISAO
       }
       return -1;
    }
  
    public void apaga(String chave) {
      int pos = busca(chave);
      if (pos != -1) {
         tab[pos].ocupado = false;
         System.out.print("-> Dado HASH[" + pos + "] apagado");
      }
      else System.out.print("Item nao encontrado");
    }
  
    public void imprime() {
      for (int i=0; i<TAM_MAX; i++)
         if ( tab[i].ocupado )
            System.out.print("\nHash[" + i + "] = " + tab[i].item);
    }
    
    public boolean cheia() {
      int i, qtde=0;
      for (i=0; i<TAM_MAX; i++)
       if( tab[i].ocupado ) qtde++;
      if (qtde == TAM_MAX) return true;
      return false;
  }
  
  }