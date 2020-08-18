<%-- 
    Document   : modelo1
    Created on : 09/05/2013, 20:09:22
    Author     : ALUNOS
--%>

<%@tag description="Tag de criação de cabeçalho e rodapé" pageEncoding="UTF-8"%>
<%@attribute name="telefone" %>
<%@attribute name="nome" %>
<!DOCTYPE html>
<html>
    <head>
        <meta http-equiv="content-Type" content="text/html; charset=UTF-8">
        <title>${telefone}</title>
    </head>
    <body>
        ${nome}
    </body>

</html>
