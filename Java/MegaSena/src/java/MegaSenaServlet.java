/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */



import java.io.IOException;
import java.io.PrintWriter;
import java.util.Calendar;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.util.Random;

/**
 *
 * @author ALUNOS
 */
@WebServlet(name = "MegaSenaServlet", urlPatterns = {"/MegaSenaServlet"})
public class MegaSenaServlet extends HttpServlet {
    @Override
protected void doGet(HttpServletRequest resquest,HttpServletResponse response)
            throws ServletException,IOException {

    
    {
        Random gerador = new Random();
 
        int numero = gerador.nextInt(10) + 1;
        int numero2 = gerador.nextInt(10) + 10;
        int numero3 = gerador.nextInt(10) + 20;
        int numero4 = gerador.nextInt(10) + 30;
        int numero5 = gerador.nextInt(10) + 40;
        int numero6 = gerador.nextInt(10) + 50;
    
     response.setContentType("text/html;charset=UTF-8");
         PrintWriter out  = response.getWriter();
         
                     
                                                          
        try {
          
            out.println("<!DOCTYPE html>");
            out.println("<html>");
            out.println("<head><title>Mega-Sena</title></head>");            
            out.println("<body><h1>Primeira Dezena: " + numero +"</h1></body>");     
            out.println("<body><h1>Segunda Dezena: " + numero2 +"</h1></body>");
            out.println("<body><h1>Terceira Dezena: " + numero3 +"</h1></body>");
            out.println("<body><h1>Quarta Dezena: " + numero4 +"</h1></body>");
            out.println("<body><h1>Quinta Dezena: " + numero5 +"</h1></body>");
            out.println("<body><h1>Sexta Dezena: " + numero6 +"</h1></body>");
            
            out.println("</html>");
        } finally {            
            out.close();
        }
    }


    }
}

        


   