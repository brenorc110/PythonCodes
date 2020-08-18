<%@ Page Title="" Language="C#" MasterPageFile="~/MasterPages/Painel.Master" AutoEventWireup="true" CodeBehind="UsuariosEditar.aspx.cs" Inherits="Sistema.Sistema.UsuariosEditar" %>
<asp:Content ID="Content3" ContentPlaceHolderID="MainContentPlaceHolder" runat="server">
    <div class="page-header">
        <h1>Editar Usuário</h1>
    </div>

    <div class="span12 center">
        <fieldset>
            <asp:Label ID="lbNome" runat="server" Text="Nome:"></asp:Label>
            <asp:TextBox ID="txNome" runat="server"></asp:TextBox>
            <br />

            <asp:Label ID="lbEmail" runat="server" Text="Email:"></asp:Label>
            <asp:TextBox ID="txEmail" runat="server"></asp:TextBox>
            <br />

            <asp:Label ID="lbSenha" runat="server" Text="Senha:"></asp:Label>
            <asp:TextBox ID="txSenha" runat="server"></asp:TextBox>
            <br />

            <asp:Label ID="lbAtivo" runat="server" Text="Ativo:"></asp:Label>
            <asp:CheckBox ID="ckAtivo" runat="server" />
            <br /><br />

            <asp:Button ID="btSalvar" runat="server" Text="Salvar"  class="btn btn-info" 
                            style="width: 100px"
                onclick="btSalvar_Click" />

            <asp:Button ID="btnCancelar" runat="server" Text="Cancelar"  class="btn" 
                            style="width: 100px" 
                onclick="btnCancelar_Click" />

            <br /><br />
        </fieldset>
    </div>
</asp:Content>
