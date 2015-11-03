Public Class Options

    Private Sub Options_FormClosed(ByVal sender As Object, ByVal e As System.Windows.Forms.FormClosedEventArgs) Handles Me.FormClosed

        Try
            CellGraph.Altura = Integer.Parse(txtAltura.Text)
            CellGraph.Largura = Integer.Parse(txtLargura.Text)
            CellGraph.fillBrush = New SolidBrush(txtCores.BackColor)
        Catch ex As Exception
            'Faz nd nao!
        End Try
    End Sub

    Private Sub Options_Load(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles MyBase.Load

        txtAltura.Text = CellGraph.Altura.ToString
        txtLargura.Text = CellGraph.Largura.ToString
        txtCores.BackColor = CellGraph.fillBrush.Color
        ColorDialog1.SolidColorOnly = True

    End Sub


    Private Sub txtCores_Click(ByVal sender As Object, ByVal e As System.EventArgs) Handles txtCores.Click

        ColorDialog1.ShowDialog()
        txtCores.BackColor = ColorDialog1.Color

    End Sub

End Class