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


@WebServlet(name = "FormDeContato", urlPatterns = {"/FormDeContato"})
public class FormDeContato extends HttpServlet {
@Override
 protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
    HttpSession session = request.getSession();
    response.setContentType("text/html;charset=UTF-8");
    PrintWriter out = response.getWriter();
    try{
            out.println("<!DOCTYPE html>");
            out.println("<html>");
            out.println("<head><title>Formulário de contato</title></head>");
            out.println("<body>");
            
            out.println("<h1>Formulário de Contato:</h1>");
            
            out.println("<form method='post'>");
            
            out.println("<label for='email'>E-mail:</label>");
            out.println("<input type='text' name='email'/>");
            out.println("<br/>");
            out.println("<label for='senha'>Senha:</label>");
            out.println("<input type='text' name='senha'/>");
            out.println("<br/>");
            out.println("<input type='submit'/>");
            out.println("</form>");
            
            out.println("</body>");
            out.println("</html>");
            } finally {            
            out.close();
        }
    }
@Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
    HttpSession session = request.getSession(true);

        response.setContentType("text/html;charset=UTF-8");
        PrintWriter out = response.getWriter();
        
        String valor_e = request.getParameter("email");
        String valor_s = request.getParameter("senha");
        session.setAttribute("email", valor_e);
        session.setAttribute("senha", valor_s);
        
        if(!"joao.silva@meuemail.com".equals(request.getParameter("email"))){
                
                response.sendRedirect("/ACVA_-_Aula_5/FormDeContato");
            } else if(!"123".equals(request.getParameter("senha"))){
                
                response.sendRedirect("/ACVA_-_Aula_5/FormDeContato");
            } else {
            out.println("<html><body>");
            out.println("<h1>E-mail: " + session.getAttribute("email") + "<h1>");
            out.println("<h2>Senha: " + session.getAttribute("senha") + "<h2>");
            out.println("</body></html>");
            }
            }
        }



   