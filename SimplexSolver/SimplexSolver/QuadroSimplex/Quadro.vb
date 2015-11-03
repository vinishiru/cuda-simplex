Public Class Quadro

    Property ColunaHeader As String() 'Array que guarda o nome das variáveis do cabeçalho das colunas do quadro
    Property LinhaHeader As String() 'Array que guarda o nome das variáveis do cabeçalho das linhas do quadro
    Property Matriz As Celula(,) 'Matriz que compoem o Quadro Simplex, este objeto é fundamental para o algoritmo.
    Property PermElemento As Celula = Nothing 'Armazena a celula do elemento permissivel encontrado no decorrer do algoritmo.
    Property PermElementoPos As Point = Nothing 'Variavel para guardar a posição do elemento permissível quando encontrado.

    Sub New(ByVal FuncaoObj As FObjetivo)


        ReDim Matriz(FuncaoObj.VariaveisArtificiais.Count, FuncaoObj.Variaveis.Count)
        ReDim ColunaHeader(FuncaoObj.Variaveis.Count) '+1 para o Bj
        ReDim LinhaHeader(FuncaoObj.VariaveisArtificiais.Count) '+1 para funcao obj

        buildHeaders(FuncaoObj)

        buildMatriz(FuncaoObj)
    End Sub

    Private Sub buildHeaders(ByVal funcaoObj As FObjetivo)

        For i As Integer = 0 To ColunaHeader.Count - 1
            If i = 0 Then
                ColunaHeader(i) = "Bj"
            Else
                ColunaHeader(i) = funcaoObj.Variaveis("X" & i).Nome
            End If
        Next

        For i As Integer = 0 To LinhaHeader.Count - 1
            If i = 0 Then
                LinhaHeader(i) = "F"
            Else
                LinhaHeader(i) = funcaoObj.VariaveisArtificiais("X" & (i + funcaoObj.Variaveis.Count)).Nome
            End If
        Next

    End Sub

    Private Sub buildMatriz(ByVal FuncaoObj As FObjetivo)

        'Linha da funcao objetivo
        Matriz(0, 0) = New Celula With {.ValorSuperior = FuncaoObj.TermoLivre}
        For i As Integer = 1 To FuncaoObj.Variaveis.Count
            Matriz(0, i) = New Celula With {.ValorSuperior = FuncaoObj.Variaveis("X" & i).Coeficiente}
        Next

        Dim linha As Integer = 1

        For Each rest As Restricao In FuncaoObj.Restricoes.Values
            ' Coluna 0 sera o termo livre
            Matriz(linha, 0) = New Celula With {.ValorSuperior = rest.TermoLivre}

            For col As Integer = 1 To FuncaoObj.Variaveis.Count
                Try
                    Matriz(linha, col) = New Celula With {.ValorSuperior = rest.Variaveis("X" & col).Coeficiente}
                Catch ex As Exception
                    Matriz(linha, col) = New Celula With {.ValorSuperior = 0}
                End Try
            Next
            linha += 1
        Next

    End Sub

    Public Function PaintQuadroSimplex(ByVal g As Graphics) As Point

        g.Clear(Color.White)


        'Pintar Coluna Headers
        For i As Integer = 0 To ColunaHeader.Length - 1
            CellGraph.PaintColHeaderGraphics(ColunaHeader(i), g, i, 0)
        Next
        'Pintar Linha Headers
        For i As Integer = 0 To LinhaHeader.Length - 1
            CellGraph.PaintRowHeaderGraphics(LinhaHeader(i), g, 0, i)
        Next

        'Pintar celulas
        For i As Integer = 0 To Matriz.GetUpperBound(0)
            For j As Integer = 0 To Matriz.GetUpperBound(1)
                CellGraph.PaintCellGraphics(Matriz(i, j), g, j, i)
            Next
        Next

        Return New Point(CellGraph.Max_X, CellGraph.Max_Y)
    End Function

    Public Sub deFlagAll()

        For i As Integer = 0 To Matriz.GetUpperBound(0)
            For j As Integer = 0 To Matriz.GetUpperBound(1)
                Matriz(i, j).Flag = False
                Matriz(i, j).FlagInferior = False
                Matriz(i, j).FlagSuperior = False
                Matriz(i, j).ValorInferior = 0.0
            Next
        Next

    End Sub

    Sub FlagPermElement(ByVal x_index As Integer, ByVal y_index As Integer)

        'Desmarcar todas
        Me.deFlagAll()

        For i As Integer = 0 To Matriz.GetUpperBound(0)
            For j As Integer = 0 To Matriz.GetUpperBound(1)
                If i = x_index Or j = y_index Then
                    Matriz(i, j).Flag = True
                End If
            Next
        Next
        'Definir elemento permissivel
        PermElemento = Matriz(x_index, y_index)
        PermElementoPos = New Point(x_index, y_index)
    End Sub

End Class
