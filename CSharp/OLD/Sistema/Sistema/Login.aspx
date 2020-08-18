<%@ Page Title="" Language="C#" MasterPageFile="~/MasterPages/Login.Master" AutoEventWireup="true" CodeBehind="Login.aspx.cs" Inherits="Sistema.Login1" %>
<asp:Content ID="ContentHeader" ContentPlaceHolderID="head" runat="server">
</asp:Content>

<asp:Content ID="Content" ContentPlaceHolderID="MainPlaceHolder" runat="server">
    <div style="width: 500px" class="center">
        <fieldset>
            <legend>Meu Sistema - Acesso Restrito</legend>
            <asp:Label ID="lbEmail" runat="server" Text="E-mail:"></asp:Label>
            <asp:TextBox ID="txEmail" runat="server" class="input-xlarge"></asp:TextBox>
            <br />

            <asp:Label ID="lbSenha" runat="server" Text="Senha:"></asp:Label>
            <asp:TextBox ID="txSenha" runat="server" TextMode="Password" class="input-xlarge"></asp:TextBox>
            <br />

            <asp:Button ID="btLogin" runat="server" Text="Login" class="btn" 
                onclick="btLogin_Click" />
        </fieldset>
   </div>

    
</asp:Content>
