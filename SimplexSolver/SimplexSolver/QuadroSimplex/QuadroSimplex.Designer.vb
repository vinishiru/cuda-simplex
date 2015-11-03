<Global.Microsoft.VisualBasic.CompilerServices.DesignerGenerated()> _
Partial Class QuadroSimplex
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
        Me.canvas = New System.Windows.Forms.Panel()
        Me.SplitContainer1 = New System.Windows.Forms.SplitContainer()
        Me.btnSolve = New System.Windows.Forms.Button()
        Me.btnStep = New System.Windows.Forms.Button()
        Me.container = New System.Windows.Forms.Panel()
        Me.lblStatus = New System.Windows.Forms.Label()
        Me.Label1 = New System.Windows.Forms.Label()
        Me.lblFase = New System.Windows.Forms.Label()
        Me.lblAlg = New System.Windows.Forms.Label()
        CType(Me.SplitContainer1, System.ComponentModel.ISupportInitialize).BeginInit()
        Me.SplitContainer1.Panel1.SuspendLayout()
        Me.SplitContainer1.Panel2.SuspendLayout()
        Me.SplitContainer1.SuspendLayout()
        Me.container.SuspendLayout()
        Me.SuspendLayout()
        '
        'canvas
        '
        Me.canvas.Anchor = CType(((System.Windows.Forms.AnchorStyles.Top Or System.Windows.Forms.AnchorStyles.Left) _
                    Or System.Windows.Forms.AnchorStyles.Right), System.Windows.Forms.AnchorStyles)
        Me.canvas.BackColor = System.Drawing.SystemColors.Window
        Me.canvas.Location = New System.Drawing.Point(0, 0)
        Me.canvas.Name = "canvas"
        Me.canvas.Size = New System.Drawing.Size(469, 103)
        Me.canvas.TabIndex = 0
        '
        'SplitContainer1
        '
        Me.SplitContainer1.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle
        Me.SplitContainer1.Dock = System.Windows.Forms.DockStyle.Fill
        Me.SplitContainer1.Location = New System.Drawing.Point(0, 0)
        Me.SplitContainer1.Name = "SplitContainer1"
        Me.SplitContainer1.Orientation = System.Windows.Forms.Orientation.Horizontal
        '
        'SplitContainer1.Panel1
        '
        Me.SplitContainer1.Panel1.AutoScroll = True
        Me.SplitContainer1.Panel1.Controls.Add(Me.btnSolve)
        Me.SplitContainer1.Panel1.Controls.Add(Me.btnStep)
        Me.SplitContainer1.Panel1.Controls.Add(Me.container)
        '
        'SplitContainer1.Panel2
        '
        Me.SplitContainer1.Panel2.Controls.Add(Me.lblStatus)
        Me.SplitContainer1.Panel2.Controls.Add(Me.Label1)
        Me.SplitContainer1.Panel2.Controls.Add(Me.lblFase)
        Me.SplitContainer1.Panel2.Controls.Add(Me.lblAlg)
        Me.SplitContainer1.Size = New System.Drawing.Size(642, 386)
        Me.SplitContainer1.SplitterDistance = 290
        Me.SplitContainer1.SplitterWidth = 5
        Me.SplitContainer1.TabIndex = 3
        '
        'btnSolve
        '
        Me.btnSolve.Location = New System.Drawing.Point(326, 248)
        Me.btnSolve.Name = "btnSolve"
        Me.btnSolve.Size = New System.Drawing.Size(75, 23)
        Me.btnSolve.TabIndex = 3
        Me.btnSolve.Text = "Resolver"
        Me.btnSolve.UseVisualStyleBackColor = True
        '
        'btnStep
        '
        Me.btnStep.Location = New System.Drawing.Point(221, 248)
        Me.btnStep.Name = "btnStep"
        Me.btnStep.Size = New System.Drawing.Size(75, 23)
        Me.btnStep.TabIndex = 2
        Me.btnStep.Text = "Avançar"
        Me.btnStep.UseVisualStyleBackColor = True
        '
        'container
        '
        Me.container.Anchor = CType(((System.Windows.Forms.AnchorStyles.Top Or System.Windows.Forms.AnchorStyles.Left) _
                    Or System.Windows.Forms.AnchorStyles.Right), System.Windows.Forms.AnchorStyles)
        Me.container.AutoScroll = True
        Me.container.BackColor = System.Drawing.SystemColors.Window
        Me.container.BorderStyle = System.Windows.Forms.BorderStyle.Fixed3D
        Me.container.Controls.Add(Me.canvas)
        Me.container.Location = New System.Drawing.Point(12, 12)
        Me.container.Name = "container"
        Me.container.Size = New System.Drawing.Size(618, 230)
        Me.container.TabIndex = 1
        '
        'lblStatus
        '
        Me.lblStatus.AutoSize = True
        Me.lblStatus.Location = New System.Drawing.Point(73, 59)
        Me.lblStatus.Name = "lblStatus"
        Me.lblStatus.Size = New System.Drawing.Size(78, 13)
        Me.lblStatus.TabIndex = 3
        Me.lblStatus.Text = "Em andamento"
        '
        'Label1
        '
        Me.Label1.AutoSize = True
        Me.Label1.Location = New System.Drawing.Point(15, 59)
        Me.Label1.Name = "Label1"
        Me.Label1.Size = New System.Drawing.Size(52, 13)
        Me.Label1.TabIndex = 2
        Me.Label1.Text = "Situação:"
        '
        'lblFase
        '
        Me.lblFase.AutoSize = True
        Me.lblFase.Location = New System.Drawing.Point(14, 35)
        Me.lblFase.Name = "lblFase"
        Me.lblFase.Size = New System.Drawing.Size(40, 13)
        Me.lblFase.TabIndex = 1
        Me.lblFase.Text = "lblFase"
        '
        'lblAlg
        '
        Me.lblAlg.AutoSize = True
        Me.lblAlg.Location = New System.Drawing.Point(14, 12)
        Me.lblAlg.Name = "lblAlg"
        Me.lblAlg.Size = New System.Drawing.Size(60, 13)
        Me.lblAlg.TabIndex = 0
        Me.lblAlg.Text = "lblAlgoritmo"
        '
        'QuadroSimplex
        '
        Me.AutoScaleDimensions = New System.Drawing.SizeF(6.0!, 13.0!)
        Me.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font
        Me.ClientSize = New System.Drawing.Size(642, 386)
        Me.Controls.Add(Me.SplitContainer1)
        Me.Name = "QuadroSimplex"
        Me.ShowIcon = False
        Me.Text = "Quadro Simplex"
        Me.SplitContainer1.Panel1.ResumeLayout(False)
        Me.SplitContainer1.Panel2.ResumeLayout(False)
        Me.SplitContainer1.Panel2.PerformLayout()
        CType(Me.SplitContainer1, System.ComponentModel.ISupportInitialize).EndInit()
        Me.SplitContainer1.ResumeLayout(False)
        Me.container.ResumeLayout(False)
        Me.ResumeLayout(False)

    End Sub
    Friend WithEvents SplitContainer1 As System.Windows.Forms.SplitContainer
    Friend WithEvents canvas As System.Windows.Forms.Panel
    Friend WithEvents container As System.Windows.Forms.Panel
    Friend WithEvents btnSolve As System.Windows.Forms.Button
    Friend WithEvents btnStep As System.Windows.Forms.Button
    Friend WithEvents lblAlg As System.Windows.Forms.Label
    Friend WithEvents lblFase As System.Windows.Forms.Label
    Friend WithEvents lblStatus As System.Windows.Forms.Label
    Friend WithEvents Label1 As System.Windows.Forms.Label
End Class
