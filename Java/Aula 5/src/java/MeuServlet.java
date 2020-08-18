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


@WebServlet(name = "MeuServlet", urlPatterns = {"/MeuServlet"})
public class MeuServlet extends HttpServlet {
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
            
            out.println("<h1>Formulário de Teste de Session</h1>");
            
            out.println("<h2>Valor da session atuial: " + session.getAttribute("chave") + "</h2>");
            out.println("<form method='post'>");
            
            out.println("<label for='valor'>Valor:</label>");
            out.println("<input type='text' name='valor'/>");
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
        
        String valor = request.getParameter("valor");
        session.setAttribute("chave", valor);
        
        response.setContentType("text/html;charset=UTF-8");
        
        out.println("<html><body>");
        out.println("<h1>Valor da session: " + session.getAttribute("chave") + "<h1>");
        out.println("</body></html>");
}
} 


   