import java.util.*; // pacote contem System.in

class HashTableLinear {
    public static void main(String[] args) {
      HashLinear tab = new HashLinear(7);
      Scanner le = new Scanner(System.in);
      String item;
  
      System.out.println("Tabela HASH - informe o valor: ");
      for (int i=0; i<7; i++){
      System.out.print("\n\nInserindo elemento " + (i+1) );
      System.out.print(" - Forneca o valor: ");
      item = le.nextLine();
      tab.insere(item);
      }
  
      System.out.print("\n\nBuscando campo\n>>> Forneca o item: ");
      item = le.nextLine();
      if ( tab.busca(item) != -1 )
         System.out.print("Item encontrado na posicao " + tab.busca(item));
      else
         System.out.print("Item nao encontrado");
      
      System.out.print("\n\nApagando campo\n>>> Forneca o item: ");
      item = le.nextLine();
      tab.apaga(item);
      
      System.out.print("\n\nImprimindo conteudo");
      tab.imprime();
  
      System.out.println("\n");
      le.close();
    }
  }