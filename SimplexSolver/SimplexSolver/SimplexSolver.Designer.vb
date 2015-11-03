<Global.Microsoft.VisualBasic.CompilerServices.DesignerGenerated()> _
Partial Class SimplexSolver
    Inherits System.Windows.Forms.Form

    'Form overrides dispose to clean up the component list.
    <System.Diagnostics.DebuggerNonUserCode()> _
    Protected Overrides Sub Dispose(ByVal disposing As Boolean)
        Try
            If disposing AndAlso components IsNot Nothing Then
                components.Dispose()
            End If
        Finally
            MyBase.Dispose(disposing)
        End Try
    End Sub

    'Required by the Windows Form Designer
    Private components As System.ComponentModel.IContainer

    'NOTE: The following procedure is required by the Windows Form Designer
    'It can be modified using the Windows Form Designer.  
    'Do not modify it using the code editor.
    <System.Diagnostics.DebuggerStepThrough()> _
    Private Sub InitializeComponent()
        Me.MenuStrip1 = New System.Windows.Forms.MenuStrip()
        Me.NovaFunçãoToolStripMenuItem = New System.Windows.Forms.ToolStripMenuItem()
        Me.NovaFunçãoToolStripMenuItem1 = New System.Windows.Forms.ToolStripMenuItem()
        Me.SairToolStripMenuItem1 = New System.Windows.Forms.ToolStripMenuItem()
        Me.FuncoesToolStripMenuItem = New System.Windows.Forms.ToolStripMenuItem()
        Me.MostrarToolStripMenuItem = New System.Windows.Forms.ToolStripMenuItem()
        Me.TesteToolStripMenuItem = New System.Windows.Forms.ToolStripMenuItem()
        Me.GroupBox1 = New System.Windows.Forms.GroupBox()
        Me.cbxExtr = New System.Windows.Forms.ComboBox()
        Me.btnDelVar = New System.Windows.Forms.Button()
        Me.Label1 = New System.Windows.Forms.Label()
        Me.btnNewVar = New System.Windows.Forms.Button()
        Me.FOGrid = New System.Windows.Forms.DataGridView()
        Me.FOHeader = New System.Windows.Forms.DataGridViewTextBoxColumn()
        Me.restbox = New System.Windows.Forms.GroupBox()
        Me.RestGrid = New System.Windows.Forms.DataGridView()
        Me.btnGenSimplex = New System.Windows.Forms.Button()
        Me.OpçõesToolStripMenuItem = New System.Windows.Forms.ToolStripMenuItem()
        Me.MenuStrip1.SuspendLayout()
        Me.GroupBox1.SuspendLayout()
        CType(Me.FOGrid, System.ComponentModel.ISupportInitialize).BeginInit()
        Me.restbox.SuspendLayout()
        CType(Me.RestGrid, System.ComponentModel.ISupportInitialize).BeginInit()
        Me.SuspendLayout()
        '
        'MenuStrip1
        '
        Me.MenuStrip1.Items.AddRange(New System.Windows.Forms.ToolStripItem() {Me.NovaFunçãoToolStripMenuItem, Me.FuncoesToolStripMenuItem})
        Me.MenuStrip1.Location = New System.Drawing.Point(0, 0)
        Me.MenuStrip1.Name = "MenuStrip1"
        Me.MenuStrip1.Size = New System.Drawing.Size(508, 24)
        Me.MenuStrip1.TabIndex = 0
        Me.MenuStrip1.Text = "MenuStrip1"
        '
        'NovaFunçãoToolStripMenuItem
        '
        Me.NovaFunçãoToolStripMenuItem.DropDownItems.AddRange(New System.Windows.Forms.ToolStripItem() {Me.NovaFunçãoToolStripMenuItem1, Me.OpçõesToolStripMenuItem, Me.SairToolStripMenuItem1})
        Me.NovaFunçãoToolStripMenuItem.Name = "NovaFunçãoToolStripMenuItem"
        Me.NovaFunçãoToolStripMenuItem.Size = New System.Drawing.Size(50, 20)
        Me.NovaFunçãoToolStripMenuItem.Text = "Menu"
        '
        'NovaFunçãoToolStripMenuItem1
        '
        Me.NovaFunçãoToolStripMenuItem1.Name = "NovaFunçãoToolStripMenuItem1"
        Me.NovaFunçãoToolStripMenuItem1.Size = New System.Drawing.Size(152, 22)
        Me.NovaFunçãoToolStripMenuItem1.Text = "Nova função..."
        '
        'SairToolStripMenuItem1
        '
        Me.SairToolStripMenuItem1.Name = "SairToolStripMenuItem1"
        Me.SairToolStripMenuItem1.Size = New System.Drawing.Size(152, 22)
        Me.SairToolStripMenuItem1.Text = "Sair"
        '
        'FuncoesToolStripMenuItem
        '
        Me.FuncoesToolStripMenuItem.DropDownItems.AddRange(New System.Windows.Forms.ToolStripItem() {Me.MostrarToolStripMenuItem, Me.TesteToolStripMenuItem})
        Me.FuncoesToolStripMenuItem.Name = "FuncoesToolStripMenuItem"
        Me.FuncoesToolStripMenuItem.Size = New System.Drawing.Size(63, 20)
        Me.FuncoesToolStripMenuItem.Text = "Funcoes"
        '
        'MostrarToolStripMenuItem
        '
        Me.MostrarToolStripMenuItem.Name = "MostrarToolStripMenuItem"
        Me.MostrarToolStripMenuItem.Size = New System.Drawing.Size(115, 22)
        Me.MostrarToolStripMenuItem.Text = "Mostrar"
        '
        'TesteToolStripMenuItem
        '
        Me.TesteToolStripMenuItem.Name = "TesteToolStripMenuItem"
        Me.TesteToolStripMenuItem.Size = New System.Drawing.Size(115, 22)
        Me.TesteToolStripMenuItem.Text = "Teste"
        '
        'GroupBox1
        '
        Me.GroupBox1.Anchor = CType(((System.Windows.Forms.AnchorStyles.Top Or System.Windows.Forms.AnchorStyles.Left) _
                    Or System.Windows.Forms.AnchorStyles.Right), System.Windows.Forms.AnchorStyles)
        Me.GroupBox1.AutoSize = True
        Me.GroupBox1.Controls.Add(Me.cbxExtr)
        Me.GroupBox1.Controls.Add(Me.btnDelVar)
        Me.GroupBox1.Controls.Add(Me.Label1)
        Me.GroupBox1.Controls.Add(Me.btnNewVar)
        Me.GroupBox1.Controls.Add(Me.FOGrid)
        Me.GroupBox1.Location = New System.Drawing.Point(13, 28)
        Me.GroupBox1.Name = "GroupBox1"
        Me.GroupBox1.Size = New System.Drawing.Size(482, 121)
        Me.GroupBox1.TabIndex = 1
        Me.GroupBox1.TabStop = False
        Me.GroupBox1.Text = "Função Objetivo"
        '
        'cbxExtr
        '
        Me.cbxExtr.Anchor = System.Windows.Forms.AnchorStyles.Left
        Me.cbxExtr.FormattingEnabled = True
        Me.cbxExtr.Location = New System.Drawing.Point(62, 80)
        Me.cbxExtr.Name = "cbxExtr"
        Me.cbxExtr.Size = New System.Drawing.Size(64, 21)
        Me.cbxExtr.TabIndex = 4
        '
        'btnDelVar
        '
        Me.btnDelVar.Anchor = System.Windows.Forms.AnchorStyles.Right
        Me.btnDelVar.Location = New System.Drawing.Point(366, 79)
        Me.btnDelVar.Name = "btnDelVar"
        Me.btnDelVar.Size = New System.Drawing.Size(110, 23)
        Me.btnDelVar.TabIndex = 3
        Me.btnDelVar.Text = "Remover variável"
        Me.btnDelVar.UseVisualStyleBackColor = True
        '
        'Label1
        '
        Me.Label1.Anchor = System.Windows.Forms.AnchorStyles.Left
        Me.Label1.AutoSize = True
        Me.Label1.Location = New System.Drawing.Point(8, 84)
        Me.Label1.Name = "Label1"
        Me.Label1.Size = New System.Drawing.Size(48, 13)
        Me.Label1.TabIndex = 2
        Me.Label1.Text = "Extremo:"
        '
        'btnNewVar
        '
        Me.btnNewVar.Anchor = System.Windows.Forms.AnchorStyles.Right
        Me.btnNewVar.Location = New System.Drawing.Point(250, 79)
        Me.btnNewVar.Name = "btnNewVar"
        Me.btnNewVar.Size = New System.Drawing.Size(110, 23)
        Me.btnNewVar.TabIndex = 1
        Me.btnNewVar.Text = "Adicionar variável"
        Me.btnNewVar.UseVisualStyleBackColor = True
        '
        'FOGrid
        '
        Me.FOGrid.AllowUserToAddRows = False
        Me.FOGrid.AllowUserToDeleteRows = False
        Me.FOGrid.Anchor = CType(((System.Windows.Forms.AnchorStyles.Top Or System.Windows.Forms.AnchorStyles.Left) _
                    Or System.Windows.Forms.AnchorStyles.Right), System.Windows.Forms.AnchorStyles)
        Me.FOGrid.BackgroundColor = System.Drawing.SystemColors.Window
        Me.FOGrid.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize
        Me.FOGrid.Columns.AddRange(New System.Windows.Forms.DataGridViewColumn() {Me.FOHeader})
        Me.FOGrid.GridColor = System.Drawing.SystemColors.Window
        Me.FOGrid.Location = New System.Drawing.Point(7, 20)
        Me.FOGrid.Name = "FOGrid"
        Me.FOGrid.RowHeadersVisible = False
        Me.FOGrid.Size = New System.Drawing.Size(469, 53)
        Me.FOGrid.TabIndex = 0
        '
        'FOHeader
        '
        Me.FOHeader.HeaderText = ""
        Me.FOHeader.Name = "FOHeader"
        Me.FOHeader.ReadOnly = True
        Me.FOHeader.Resizable = System.Windows.Forms.DataGridViewTriState.[False]
        Me.FOHeader.Width = 40
        '
        'restbox
        '
        Me.restbox.AutoSize = True
        Me.restbox.Controls.Add(Me.RestGrid)
        Me.restbox.Location = New System.Drawing.Point(13, 155)
        Me.restbox.Name = "restbox"
        Me.restbox.Size = New System.Drawing.Size(483, 249)
        Me.restbox.TabIndex = 2
        Me.restbox.TabStop = False
        Me.restbox.Text = "Restrições"
        '
        'RestGrid
        '
        Me.RestGrid.Anchor = CType((((System.Windows.Forms.AnchorStyles.Top Or System.Windows.Forms.AnchorStyles.Bottom) _
                    Or System.Windows.Forms.AnchorStyles.Left) _
                    Or System.Windows.Forms.AnchorStyles.Right), System.Windows.Forms.AnchorStyles)
        Me.RestGrid.BackgroundColor = System.Drawing.SystemColors.Window
        Me.RestGrid.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize
        Me.RestGrid.GridColor = System.Drawing.SystemColors.Window
        Me.RestGrid.Location = New System.Drawing.Point(7, 20)
        Me.RestGrid.Name = "RestGrid"
        Me.RestGrid.Size = New System.Drawing.Size(469, 219)
        Me.RestGrid.TabIndex = 0
        '
        'btnGenSimplex
        '
        Me.btnGenSimplex.Anchor = System.Windows.Forms.AnchorStyles.Bottom
        Me.btnGenSimplex.Location = New System.Drawing.Point(189, 410)
        Me.btnGenSimplex.Name = "btnGenSimplex"
        Me.btnGenSimplex.Size = New System.Drawing.Size(128, 23)
        Me.btnGenSimplex.TabIndex = 3
        Me.btnGenSimplex.Text = "Gerar quadro Simplex"
        Me.btnGenSimplex.UseVisualStyleBackColor = True
        '
        'OpçõesToolStripMenuItem
        '
        Me.OpçõesToolStripMenuItem.Name = "OpçõesToolStripMenuItem"
        Me.OpçõesToolStripMenuItem.Size = New System.Drawing.Size(152, 22)
        Me.OpçõesToolStripMenuItem.Text = "Opções..."
        '
        'SimplexSolver
        '
        Me.AutoScaleDimensions = New System.Drawing.SizeF(6.0!, 13.0!)
        Me.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font
        Me.AutoSize = True
        Me.ClientSize = New System.Drawing.Size(508, 445)
        Me.Controls.Add(Me.btnGenSimplex)
        Me.Controls.Add(Me.restbox)
        Me.Controls.Add(Me.GroupBox1)
        Me.Controls.Add(Me.MenuStrip1)
        Me.MainMenuStrip = Me.MenuStrip1
        Me.MaximizeBox = False
        Me.Name = "SimplexSolver"
        Me.ShowIcon = False
        Me.Text = "Simplex Solver"
        Me.MenuStrip1.ResumeLayout(False)
        Me.MenuStrip1.PerformLayout()
        Me.GroupBox1.ResumeLayout(False)
        Me.GroupBox1.PerformLayout()
        CType(Me.FOGrid, System.ComponentModel.ISupportInitialize).EndInit()
        Me.restbox.ResumeLayout(False)
        CType(Me.RestGrid, System.ComponentModel.ISupportInitialize).EndInit()
        Me.ResumeLayout(False)
        Me.PerformLayout()

    End Sub
    Friend WithEvents MenuStrip1 As System.Windows.Forms.MenuStrip
    Friend WithEvents NovaFunçãoToolStripMenuItem As System.Windows.Forms.ToolStripMenuItem
    Friend WithEvents NovaFunçãoToolStripMenuItem1 As System.Windows.Forms.ToolStripMenuItem
    Friend WithEvents SairToolStripMenuItem1 As System.Windows.Forms.ToolStripMenuItem
    Friend WithEvents GroupBox1 As System.Windows.Forms.GroupBox
    Friend WithEvents btnNewVar As System.Windows.Forms.Button
    Friend WithEvents FOGrid As System.Windows.Forms.DataGridView
    Friend WithEvents restbox As System.Windows.Forms.GroupBox
    Friend WithEvents RestGrid As System.Windows.Forms.DataGridView
    Friend WithEvents btnGenSimplex As System.Windows.Forms.Button
    Friend WithEvents FOHeader As System.Windows.Forms.DataGridViewTextBoxColumn
    Friend WithEvents FuncoesToolStripMenuItem As System.Windows.Forms.ToolStripMenuItem
    Friend WithEvents MostrarToolStripMenuItem As System.Windows.Forms.ToolStripMenuItem
    Friend WithEvents btnDelVar As System.Windows.Forms.Button
    Friend WithEvents Label1 As System.Windows.Forms.Label
    Friend WithEvents cbxExtr As System.Windows.Forms.ComboBox
    Friend WithEvents TesteToolStripMenuItem As System.Windows.Forms.ToolStripMenuItem
    Friend WithEvents OpçõesToolStripMenuItem As System.Windows.Forms.ToolStripMenuItem

End Class
