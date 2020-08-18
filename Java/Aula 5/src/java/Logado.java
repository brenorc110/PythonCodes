/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

import java.io.IOException;
import java.io.PrintWriter;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;


/**
 *
 * @author ALUNOS
 */
@WebServlet(name = "Logado", urlPatterns = {"/Logado"})
public class Logado extends HttpServlet {
    
    @Override
 protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
    HttpSession session = request.getSession(true);
    response.setContentType("text/html;charset=UTF-8");
    PrintWriter out = response.getWriter();
    response.setContentType("text/html;charset=UTF-8");
    try{
          if(!"joao.silva@meuemail.com".equals(request.getParameter("email"))){
                
                response.sendRedirect("/ACVA_-_Aula_5/FormDeContato");
            } else if(!"123".equals(request.getParameter("senha"))){
                
                response.sendRedirect("/ACVA_-_Aula_5/FormDeContato");
            }else{
            out.println("<html><body>");
            out.println("<h1>E-mail logado: " + session.getAttribute("email") + "<h1>");
            out.println("<h2>Senha logado: " + session.getAttribute("senha") + "<h2>");
            out.println("</body></html>");
            }
            } finally {            
            out.close();
        }
    }

       
}



   