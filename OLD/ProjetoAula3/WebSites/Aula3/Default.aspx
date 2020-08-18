<%@ Page Title="" Language="C#" MasterPageFile="~/Site.master" AutoEventWireup="true" CodeFile="Default.aspx.cs" Inherits="_Default" %>

<asp:Content ID="Content1" ContentPlaceHolderID="head" Runat="Server">
</asp:Content>
<asp:Content ID="Content2" ContentPlaceHolderID="ContentPlaceHolder1" Runat="Server">
    <p>
        <asp:TextBox ID="txtNome" runat="server" 
            style="z-index: 1; left: 55px; top: 70px; position: absolute; width: 129px; height: 21px"></asp:TextBox>
        <br />
    </p>
    <p>
        <asp:Label ID="lblAutor" runat="server" 
            style="z-index: 1; left: 11px; top: 70px; position: absolute; width: 45px" 
            Text="Autor:"></asp:Label>
        <asp:Button ID="btnLer" runat="server" onclick="btnLer_Click" 
            style="z-index: 1; left: 206px; top: 64px; position: absolute" 
            Text="Ler Cookie" />
    </p>
    <p>
        <asp:TextBox ID="txtTitulo" runat="server" 
            style="z-index: 1; left: 55px; top: 107px; position: absolute"></asp:TextBox>
        <asp:Label ID="lblTitulo" runat="server" 
            style="z-index: 1; left: 12px; top: 110px; position: absolute; width: 38px" 
            Text="Título:"></asp:Label>
        <asp:Button ID="btnGravar" runat="server" onclick="btnGravar_Click" 
            style="z-index: 1; left: 206px; top: 102px; position: absolute; width: 96px" 
            Text="Gravar Cookie" />
    </p>
    <asp:Button ID="btnMatar" runat="server" onclick="btnMatar_Click" 
        style="z-index: 1; left: 213px; top: 148px; position: absolute; width: 83px" 
        Text="Matar Cookie" />
    <p>
        &nbsp;</p>
    <p>
        <asp:Label ID="lblMensagem" runat="server" ForeColor="Red" 
            style="z-index: 1; left: 13px; top: 259px; position: absolute; height: 21px; width: 113px; right: 1163px"></asp:Label>
        <asp:Label ID="lblSaidaTitulo" runat="server" 
            style="z-index: 1; left: 13px; top: 232px; position: absolute; height: 21px; width: 113px"></asp:Label>
        <asp:Label ID="lblSaidaAutor" runat="server" 
            style="z-index: 1; left: 13px; top: 200px; position: absolute; height: 21px; width: 113px"></asp:Label>
    </p>
    <p>
        &nbsp;</p>
    <p>
    </p>
    <p>
    </p>
</asp:Content>

