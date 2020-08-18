/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Set;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/**
 *
 * @author alunos
 */
@WebServlet(name = "Sorteio", urlPatterns = {"/Sorteio"})

public class Sorteio extends HttpServlet {
    
 @Override
    protected void doGet(HttpServletRequest resquest,HttpServletResponse response)
            throws ServletException,IOException {

    
    {
        Random gerador = new Random();
 
        int numero = gerador.nextInt(10) + 1;
        int numero2 = gerador.nextInt(10) + 11;
        int numero3 = gerador.nextInt(10) + 21;
        int numero4 = gerador.nextInt(10) + 31;
        int numero5 = gerador.nextInt(10) + 41;
        int numero6 = gerador.nextInt(10) + 51;
    
     response.setContentType("text/html;charset=UTF-8");
         PrintWriter out  = response.getWriter();
         
                     
                                                          
        try {
          
            out.println("<!DOCTYPE html>");
            out.println("<html>");
            out.println("<head><title>Simulação de Sorteio Mega-Sena:</title></head>");            
            out.println("<h1>" + numero +"</h1>");     
            out.println("<h1>" + numero2 +"</h1>");
            out.println("<h1>" + numero3 +"</h1>");
            out.println("<h1>" + numero4 +"</h1>");
            out.println("<h1>" + numero5 +"</h1>");
            out.println("<h1>" + numero6 +"</h1></body>");
            
            out.println("</html>");
        } finally {            
            out.close();
        }
    }


    }
}
        
        
    


