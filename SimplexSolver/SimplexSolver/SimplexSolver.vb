''' <summary>
''' Este aplicativo implementa o método Simplex de otimização, utilizando a implementação descrita pelo professor Dr. Petr Iakovlevitch Ekel.
''' Não é permitido utilizar este software para uso comercial sem consentimento do autor.
''' </summary>
''' <remarks>
''' Autor: Vinícius Oliveira e Silva
''' PPGEE: Programa de pós-graduação em engenharia elétrica - PUC-MG
''' </remarks>

Public Class SimplexSolver

    Private FO As FObjetivo

    Private Sub Form1_Load(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles MyBase.Load
        FO = New FObjetivo
        resetFOGrid()

        cbxExtr.Items.Add(FObjetivo.Extremo.Max)
        cbxExtr.Items.Add(FObjetivo.Extremo.Min)
    End Sub

    Private Sub NovaFunçãoToolStripMenuItem1_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles NovaFunçãoToolStripMenuItem1.Click
        FO = New FObjetivo
        resetFOGrid()
        resetRestGrid()

        'Reativar controles
        btnDelVar.Enabled = True
        btnNewVar.Enabled = True
        cbxExtr.Enabled = True
    End Sub

    Private Sub SairToolStripMenuItem1_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles SairToolStripMenuItem1.Click
        Me.Close()
    End Sub

    Private Sub btnNewVar_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles btnNewVar.Click
        Dim varAux As Variavel = FO.createNewVariavel

        'Adicionar variavel no grid da FO
        FOGrid.Columns.Add(New DataGridViewTextBoxColumn With {.ReadOnly = False, .HeaderText = varAux.Nome, .Width = 40})

        'Atualizar grid das restriçoes
        AtualizarRestGrid(varAux.Nome)
    End Sub

    Private Sub btnDelVar_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles btnDelVar.Click

        FOGrid.Columns.RemoveAt(FOGrid.Columns.Count - 1)
        FO.deleteVariavel()
        RestGrid.Columns.RemoveAt(RestGrid.Columns.Count - 3)

        For Each rest As Restricao In FO.Restricoes.Values
            rest.deleteVariavel()
        Next

    End Sub

    Private Sub AtualizarRestGrid(ByVal varNome As String)

        'Se for a primeira coluna, devo colocar a coluna do relacionamento e a coluna do termo livre
        If RestGrid.ColumnCount = 0 Then

            'Adicionar coluna do relacionamento
            RestGrid.Columns.Add(New DataGridViewComboBoxColumn With {.ReadOnly = False, .ValueType = GetType(Restricao.Relacionamento), .HeaderText = "Relacionamento", .Width = 90, .DataSource = [Enum].GetValues(GetType(Restricao.Relacionamento))})
            'Adicionar coluna do termo livres
            RestGrid.Columns.Add(New DataGridViewTextBoxColumn With {.ReadOnly = False, .HeaderText = "Termo livre", .Width = 85})

        End If

        RestGrid.Columns.Insert(RestGrid.Columns.Count - 2, New DataGridViewTextBoxColumn With {.ReadOnly = False, .HeaderText = varNome, .Width = 40})

    End Sub

    Private Sub resetFOGrid()

        FOGrid.Rows.Clear()

        'remover todas as colunas menos a primeira
        While FOGrid.Columns.Count > 1
            FOGrid.Columns.RemoveAt(1)
        End While

        'Adicionar F(x) no grid do FO
        FOGrid.Rows.Add("F(x)=")

    End Sub

    Private Sub resetRestGrid()
        RestGrid.Rows.Clear()
        RestGrid.Columns.Clear()
    End Sub

    Private Sub FOGrid_CellEndEdit(ByVal sender As Object, ByVal e As System.Windows.Forms.DataGridViewCellEventArgs) Handles FOGrid.CellEndEdit

        'Verificar qual coluna foi editada
        Dim varName As String = FOGrid.Columns(e.ColumnIndex).HeaderText
        Dim varValue As Double = Double.Parse(FOGrid.Rows(e.RowIndex).Cells(e.ColumnIndex).Value)

        FO.setVariavelValue(varName, varValue)
    End Sub

    Private Sub RestGrid_CellEndEdit(ByVal sender As Object, ByVal e As System.Windows.Forms.DataGridViewCellEventArgs) Handles RestGrid.CellEndEdit

        'Verificar qual coluna foi editada
        Dim varName As String = String.Empty
        Dim rel As Restricao.Relacionamento = Nothing
        Dim varValue As Double

        Try
            If e.ColumnIndex = RestGrid.Columns.Count - 2 Then
                Try
                    'Coluna do relacionamento
                    rel = RestGrid.Rows(e.RowIndex).Cells(e.ColumnIndex).Value
                    FO.setRestricaoRelacionamentoValue(e.RowIndex, rel)
                Catch ex As Exception
                    MsgBox("Erro ao tentar atribuir relacionamento a restrição.", MsgBoxStyle.Critical)
                    Throw ex
                End Try

            ElseIf e.ColumnIndex = RestGrid.Columns.Count - 1 Then
                Try
                    'Coluna do termo livre
                    varValue = Double.Parse(RestGrid.Rows(e.RowIndex).Cells(e.ColumnIndex).Value)
                    FO.setRestricaoTermoLivreValue(e.RowIndex, varValue)
                Catch ex As Exception
                    MsgBox("Erro ao tentar atribuir valor para termo livre.", MsgBoxStyle.Critical)
                    Throw ex
                End Try

            Else
                Try
                    'Colunas das variaveis
                    varName = RestGrid.Columns(e.ColumnIndex).HeaderText
                    varValue = Double.Parse(RestGrid.Rows(e.RowIndex).Cells(e.ColumnIndex).Value.ToString.Replace(".", ","))
                    FO.setRestricaoVariavelValue(e.RowIndex, varName, varValue)
                Catch ex As Exception
                    MsgBox("Erro ao tentar atribuir valor para variavel.", MsgBoxStyle.Critical)
                    Throw ex
                End Try

            End If
        Catch ex As Exception
            RestGrid.Rows(e.RowIndex).Cells(e.ColumnIndex).Value = Nothing
        End Try
    End Sub

    Private Function validationProcess() As Boolean


        Try
            Dim rel As Object = Nothing

            'Validar valores se todos os relacionamentos das restricoes foram setados
            For Each rest As DataGridViewRow In RestGrid.Rows
                If Not rest.IsNewRow Then
                    rel = rest.Cells(rest.Cells.Count - 2).Value
                    If rel Is Nothing Then
                        Throw New Exception("Valor de relacionamento não setado na restrição número " & rest.Index + 1 & ".")
                    End If
                End If
            Next
        Catch ex As Exception
            MsgBox(ex.Message, MsgBoxStyle.Exclamation)
            Return False
        End Try

        Return True
    End Function

    Private Sub btnGenSimplex_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles btnGenSimplex.Click

        Dim quadro As QuadroSimplex = Nothing

        If validationProcess() Then
            'A normalização das funções (Objetivo e restrições) acontece abaixo.
            FO.normalizar()
            quadro = New QuadroSimplex(FO)

            'Travar todos os controles
            btnDelVar.Enabled = False
            btnNewVar.Enabled = False
            cbxExtr.Enabled = False

            'Mostrar tela do quadro simplex
            quadro.ShowDialog()

        End If

    End Sub

    Private Sub MostrarToolStripMenuItem_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles MostrarToolStripMenuItem.Click

        Dim str As String = String.Empty

        str = FO.ToString & vbCrLf

        For Each rest As Restricao In FO.Restricoes.Values
            str += rest.ToString & vbCrLf
        Next

        MsgBox(str)

    End Sub

    Private Sub cbxExtr_SelectedIndexChanged(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles cbxExtr.SelectedIndexChanged
        FO.Extr = cbxExtr.SelectedItem
    End Sub

    Private Sub OpçõesToolStripMenuItem_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles OpçõesToolStripMenuItem.Click
        Options.ShowDialog()
    End Sub
End Class
