/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package mongodb;

import com.mongodb.BasicDBObject;
import com.mongodb.DB;
import com.mongodb.DBCollection;
import com.mongodb.DBCursor;
import java.util.logging.Level;
import java.util.logging.Logger;
import com.mongodb.MongoClient;
import com.mongodb.client.FindIterable;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoCursor;
import com.mongodb.client.MongoDatabase;
import static com.mongodb.client.model.Filters.eq;
import org.bson.Document;
/**
 *
 * @author breno
 */
public class MongoDB{

    private MongoClient mongoDB;
    private MongoDatabase db;
    public String listarDocumento(String nomeColecao){
        Logger mongoLogger  = Logger.getLogger("org.mongodb.driver");
        mongoLogger.setLevel(Level.SEVERE);
        this.mongoDB = new MongoClient("localhost", 27017);
        this.db = mongoDB.getDatabase("admin");
        MongoCollection<Document> collection = db.getCollection(nomeColecao);
        String resposta = "";
        FindIterable<Document> cursor = collection.find();
        for (Document doc: cursor){
            resposta+=doc.toJson();
            //System.out.println(doc.getString("cpf"));
            //System.out.println("doc.toJson()");
    }
    return resposta;
    }
    public void inserirDocumento(String nomeColecao){
        Logger mongoLogger  = Logger.getLogger("org.mongodb.driver");
        mongoLogger.setLevel(Level.SEVERE);
        this.mongoDB = new MongoClient("localhost", 27017);
        this.db = mongoDB.getDatabase("admin");
        MongoCollection<Document> collection = db.getCollection(nomeColecao);
        Document doc  = new Document("cpf","12345678911").append("rg","123456789").append("nome","Breno").append("curso","Ciencia");
        collection.insertOne(doc);
    }
    public void localizarDocumento(String nome, String nomeColecao){
        Logger mongoLogger  = Logger.getLogger("org.mongodb.driver");
        mongoLogger.setLevel(Level.SEVERE);
        this.mongoDB = new MongoClient("localhost", 27017);
        this.db = mongoDB.getDatabase("admin");
        MongoCollection<Document> collection = db.getCollection(nomeColecao);
        try(MongoCursor<Document> cursor = collection.find(eq("nome", nome)).iterator()){
            while(cursor.hasNext()){
                Document atual= cursor.next();
                System.out.println(atual.toJson());
            }
        }
    }
    public void alteraAluno(String nomeAntigo, String nomeNovo, String nomeColecao){
        Logger mongoLogger  = Logger.getLogger("org.mongodb.driver");
        mongoLogger.setLevel(Level.SEVERE);
        this.mongoDB = new MongoClient("localhost", 27017);
        this.db = mongoDB.getDatabase("admin");
        MongoCollection<Document> collection = db.getCollection(nomeColecao);
        collection.updateOne(eq("nome", nomeAntigo), new Document("$set", new Document("nome", nomeNovo)));
    }
    public void removerAluno(String nome, String nomeColecao){
        Logger mongoLogger  = Logger.getLogger("org.mongodb.driver");
        mongoLogger.setLevel(Level.SEVERE);
        this.mongoDB = new MongoClient("localhost", 27017);
        this.db = mongoDB.getDatabase("admin");
        MongoCollection<Document> collection = db.getCollection(nomeColecao);
        collection.deleteOne(eq("nome", nome));
    }
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        MongoDB mongo = new MongoDB();
        //mongo.inserirDocumento("Alunos");
        mongo.localizarDocumento("Breno", "Alunos");
        mongo.alteraAluno("Breno", "Brenno", "Alunos");
        mongo.removerAluno("Brenno", "Alunos");
        System.out.println(mongo.listarDocumento("Alunos"));
    }
    
}
