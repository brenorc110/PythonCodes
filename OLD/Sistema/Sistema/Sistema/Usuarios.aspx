<%@ Page Title="" Language="C#" MasterPageFile="~/MasterPages/Painel.Master" AutoEventWireup="true" CodeBehind="Usuarios.aspx.cs" Inherits="Sistema.Sistema.Usuarios" %>
<asp:Content ID="Content3" ContentPlaceHolderID="MainContentPlaceHolder" runat="server">
    <fieldset>
        <legend>Cadastro de Usuários</legend>
        <div class="row">
            <div class="span10">
                <asp:Label ID="lbNome" runat="server" Text="Nome:"></asp:Label>
                <asp:TextBox ID="txNome" runat="server"></asp:TextBox>
                <br />

                <asp:Label ID="lbEmail" runat="server" Text="Email:"></asp:Label>
                <asp:TextBox ID="txEmail" runat="server"></asp:TextBox>
                <br />

                <asp:Label ID="lbSenha" runat="server" Text="Senha:"></asp:Label>
                <asp:TextBox ID="txSenha" runat="server"></asp:TextBox>
                <br />
            </div>
            <div class="span2" style="text-align: center">
                <asp:Button ID="btNovo" runat="server" Text="Novo" class="btn btn-inverse" style="width: 100px" />
                <br /><br />
            
                <asp:Button ID="btSalvar" runat="server" Text="Salvar"  class="btn btn-info" style="width: 100px" />
                <br /><br />

                <asp:Button ID="btApagar" runat="server" Text="Apagar"  class="btn btn-danger" style="width: 100px" />
                <br />
            </div>
        </div>

        <hr style="border-top: 1px solid #999" />

        <asp:GridView ID="gvRegistros" runat="server" BorderStyle="None" 
            CssClass="table">
            <Columns>
                <asp:BoundField DataField="id" HeaderText="Cód." />
                <asp:BoundField DataField="nome" HeaderText="Nome" />
                <asp:BoundField DataField="email" HeaderText="E-mail" />
                <asp:BoundField DataField="ativo" HeaderText="Ativo" />
                
                <asp:HyperLinkField DataNavigateUrlFields="id" 
                    DataNavigateUrlFormatString="Usuarios.aspx?id={0}" HeaderText="Editar" />
            </Columns>
        </asp:GridView>
    </fieldset>
</asp:Content>
