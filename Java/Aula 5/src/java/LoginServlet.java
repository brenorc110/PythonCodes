import java.io.IOException;
import java.io.PrintWriter;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;


@WebServlet( urlPatterns = {"/LoginServlet"})
public class LoginServlet extends HttpServlet {
@Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        
        response.setContentType("text/html;charset=UTF-8");
        PrintWriter out = response.getWriter();
        try {
           out.println("<!DOCTYPE html>");
           out.println("<html>");
           out.println("<head><title>Formulário Teste de Session</title></head>");
           out.println("<body>");
           out.println("<h1>Formulário Teste de Session</h1>");
           out.println("<form method='post'>");
           out.println("<label for='email'>e-Mail:</label>");
           out.println("<input type='text' name='email'/>");
           out.println("<label for='senha'>Senha:</label>");
           out.println("<input type='text' name='senha'/>");
           out.println("<br/>");
           out.println("<input type='submit' text='enviar'/>");
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
       try {
           String email = request.getParameter("email");
           String senha = request.getParameter("senha");
           
           String[] eMails = new String[2];
           String[] senhas = new String[2];
           eMails[0] = "joao.silva@meuemail.com";
           senhas[0] = "123";
           eMails[1] = "maria.silveira@meuemail.com";
           senhas[1] = "456";
           
           for (int i = 0; i < eMails.length; i++) {
               
           if (eMails[i].equals(email) && senhas[i].equals(senha)) {
                  session.setAttribute("ID","2134");
                  session.setAttribute("EMAIL",email);
                  response.sendRedirect("PagesServlet");
                  break;
           }
           else           {
           session.setAttribute("ID","0");
           out.println("<!DOCTYPE html>");
           out.println("<html>");
           out.println("<head>");
           out.println("<title>Login incorreto</title>"); 
           out.println("</head>");
           out.println("<body>");
           out.println("</HR>");
           out.println("<p>Para voce acessar ao invés de digitar o email \"" + eMails[i] + "\"");
           out.println(" e a senha \"" + senhas[i] + "\"");
           out.println("digite \"myemail@domain.com\"");
           out.println(" e a senha \"mypass\"</p></br>");
           out.println("</HR>");
           out.println("E-Mail ou Senha incorreto!");
           out.println("<a href=\"LoginServlet\">Voltar!</a>");
           out.println("</body>");
           out.println("</html>");
                           }
           }
       } finally {            
            out.close();
        }
          
}
}