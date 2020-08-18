namespace UsandoListas
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.labelCpnj = new System.Windows.Forms.Label();
            this.boxCpnj = new System.Windows.Forms.TextBox();
            this.labelDescricao = new System.Windows.Forms.Label();
            this.boxDescricao = new System.Windows.Forms.TextBox();
            this.boxRazaoSocial = new System.Windows.Forms.TextBox();
            this.labelRazaoSocial = new System.Windows.Forms.Label();
            this.button1 = new System.Windows.Forms.Button();
            this.button2 = new System.Windows.Forms.Button();
            this.dataGridView1 = new System.Windows.Forms.DataGridView();
            ((System.ComponentModel.ISupportInitialize)(this.dataGridView1)).BeginInit();
            this.SuspendLayout();
            // 
            // labelCpnj
            // 
            this.labelCpnj.AutoSize = true;
            this.labelCpnj.Location = new System.Drawing.Point(13, 13);
            this.labelCpnj.Name = "labelCpnj";
            this.labelCpnj.Size = new System.Drawing.Size(34, 13);
            this.labelCpnj.TabIndex = 0;
            this.labelCpnj.Text = "CPNJ";
            // 
            // boxCpnj
            // 
            this.boxCpnj.Location = new System.Drawing.Point(54, 13);
            this.boxCpnj.Name = "boxCpnj";
            this.boxCpnj.Size = new System.Drawing.Size(100, 20);
            this.boxCpnj.TabIndex = 1;
            // 
            // labelDescricao
            // 
            this.labelDescricao.AutoSize = true;
            this.labelDescricao.Location = new System.Drawing.Point(160, 16);
            this.labelDescricao.Name = "labelDescricao";
            this.labelDescricao.Size = new System.Drawing.Size(58, 13);
            this.labelDescricao.TabIndex = 2;
            this.labelDescricao.Text = "Descrição:";
            // 
            // boxDescricao
            // 
            this.boxDescricao.Location = new System.Drawing.Point(224, 16);
            this.boxDescricao.Name = "boxDescricao";
            this.boxDescricao.Size = new System.Drawing.Size(100, 20);
            this.boxDescricao.TabIndex = 3;
            // 
            // boxRazaoSocial
            // 
            this.boxRazaoSocial.Location = new System.Drawing.Point(92, 55);
            this.boxRazaoSocial.Name = "boxRazaoSocial";
            this.boxRazaoSocial.Size = new System.Drawing.Size(299, 20);
            this.boxRazaoSocial.TabIndex = 4;
            // 
            // labelRazaoSocial
            // 
            this.labelRazaoSocial.AutoSize = true;
            this.labelRazaoSocial.Location = new System.Drawing.Point(13, 58);
            this.labelRazaoSocial.Name = "labelRazaoSocial";
            this.labelRazaoSocial.Size = new System.Drawing.Size(73, 13);
            this.labelRazaoSocial.TabIndex = 5;
            this.labelRazaoSocial.Text = "Razão Social:";
            // 
            // button1
            // 
            this.button1.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.button1.Location = new System.Drawing.Point(92, 110);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(277, 23);
            this.button1.TabIndex = 6;
            this.button1.Text = "Add à lista";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_OnClick);
            // 
            // button2
            // 
            this.button2.Font = new System.Drawing.Font("Microsoft Sans Serif", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.button2.Location = new System.Drawing.Point(12, 151);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(120, 23);
            this.button2.TabIndex = 7;
            this.button2.Text = "Atualizar DataGrid";
            this.button2.UseVisualStyleBackColor = true;
            this.button2.Click += new System.EventHandler(this.button2_Onclick);
            // 
            // dataGridView1
            // 
            this.dataGridView1.AllowUserToOrderColumns = true;
            this.dataGridView1.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.dataGridView1.Location = new System.Drawing.Point(0, 180);
            this.dataGridView1.Name = "dataGridView1";
            this.dataGridView1.Size = new System.Drawing.Size(422, 211);
            this.dataGridView1.TabIndex = 8;
            this.dataGridView1.CellStateChanged += new System.Windows.Forms.DataGridViewCellStateChangedEventHandler(this.dataGridView1_CellStateChanged);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(423, 392);
            this.Controls.Add(this.dataGridView1);
            this.Controls.Add(this.button2);
            this.Controls.Add(this.button1);
            this.Controls.Add(this.labelRazaoSocial);
            this.Controls.Add(this.boxRazaoSocial);
            this.Controls.Add(this.boxDescricao);
            this.Controls.Add(this.labelDescricao);
            this.Controls.Add(this.boxCpnj);
            this.Controls.Add(this.labelCpnj);
            this.Name = "Form1";
            this.Text = "Form1";
            ((System.ComponentModel.ISupportInitialize)(this.dataGridView1)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label labelCpnj;
        private System.Windows.Forms.TextBox boxCpnj;
        private System.Windows.Forms.Label labelDescricao;
        private System.Windows.Forms.TextBox boxDescricao;
        private System.Windows.Forms.TextBox boxRazaoSocial;
        private System.Windows.Forms.Label labelRazaoSocial;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.Button button2;
        private System.Windows.Forms.DataGridView dataGridView1;
    }
}

