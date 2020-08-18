

import java.io.IOException;
import java.io.PrintWriter;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
//import javax.swing.JOptionPane;

@WebServlet(name = "Get", urlPatterns = {"/Get"})
public class Get extends HttpServlet {

    
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        response.setContentType("text/html;charset=UTF-8");
        try (PrintWriter out = response.getWriter()) {
            /* TODO output your page here. You may use following sample code. */
            out.println("<!DOCTYPE html>");
            out.println("<html>");
            out.println("<head><title>Formulário de contato</title></head>");
            out.println("<body>");
            
            out.println("<h1>Formulário de Contato</h1>");
            
            out.println("<form method='post'>");
            out.println("<label for='nome'>Nome:</label>");
            out.println("<input type='text' name='nome'/>");
            out.println("<br/>");
            out.println("<label for='telefone'>Telefone:</label>");
            out.println("<input type='text' name='telefone'/>");
            out.println("<br/>");
            out.println("<label for='mensagem'>Mensagem:</label>");
            out.println("<textarea name='mensagem' rows='10' cols='30'></textarea>");
            out.println("<br/>");
            out.println("<input type='submit'/>");
            out.println("</form>");
            
            out.println("</body>");
            out.println("</html>");
        }
    }
     @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        response.setContentType("text/html;charset=UTF-8");
        PrintWriter out = response.getWriter();
       
        String nome = request.getParameter("nome");
        String telefone = request.getParameter("telefone");
        String mensagem = request.getParameter("mensagem");
        
        if(("".equals(nome)) || ("".equals(telefone)) || ("".equals(mensagem))){
            out.println("Preencha todos os campos!");
            out.println("<a href=\'Get\'>Clique pra voltar</a></br>");
        } else {
        response.setContentType("text/html;charset=UTF-8");
        out.println("Nome : " + nome);
        out.println("Telefone : " + telefone);
        out.println("Mensagem : " + mensagem);
        out.println("OK! Obrigado por preencher !");
        }
}

    
}