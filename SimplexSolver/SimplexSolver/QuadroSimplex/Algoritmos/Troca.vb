''' <summary>
''' Classe que representa todos os passos do Algoritmo de Troca do Simplex.
''' </summary>
''' <remarks></remarks>
Public Class Troca

    Private Shared alfa As Double = Nothing

    Shared Function AT_Fase1(ByVal quadro As Quadro) As Automato.Estado

        'Calcular o inverso do elemento permissivel
        alfa = 1 / quadro.PermElemento.ValorSuperior

        quadro.PermElemento.ValorInferior = alfa

        Return Automato.Estado.AT_Fase2

    End Function

    Shared Function AT_Fase2(ByVal quadro As Quadro) As Automato.Estado

        Dim auxValue As Double = Nothing
        Dim auxCell As Celula = Nothing

        For i As Integer = 0 To quadro.Matriz.GetUpperBound(1)

            auxCell = quadro.Matriz(quadro.PermElementoPos.X, i)
            If auxCell IsNot quadro.PermElemento Then
                auxValue = auxCell.ValorSuperior * alfa
                auxCell.ValorInferior = auxValue
            End If
        Next

        Return Automato.Estado.AT_Fase3
    End Function

    Shared Function AT_Fase3(ByVal quadro As Quadro) As Automato.Estado

        Dim auxValue As Double = Nothing
        Dim auxCell As Celula = Nothing

        For i As Integer = 0 To quadro.Matriz.GetUpperBound(0)
            auxCell = quadro.Matriz(i, quadro.PermElementoPos.Y)
            If auxCell IsNot quadro.PermElemento Then
                auxValue = auxCell.ValorSuperior * (alfa * -1)
                auxCell.ValorInferior = auxValue
            End If

        Next

        Return Automato.Estado.AT_Fase4
    End Function

    Shared Function AT_Fase4(ByVal quadro As Quadro) As Automato.Estado

        Dim auxValue As Double = Nothing
        Dim auxCell As Celula = Nothing

        For i As Integer = 0 To quadro.Matriz.GetUpperBound(0)
            For j As Integer = 0 To quadro.Matriz.GetUpperBound(1)
                auxCell = quadro.Matriz(i, j)
                If auxCell IsNot quadro.PermElemento Then
                    If i = quadro.PermElementoPos.X Then
                        auxCell.FlagSuperior = True
                    ElseIf j = quadro.PermElementoPos.Y Then
                        auxCell.FlagInferior = True
                    End If
                End If
            Next
        Next

        Return Automato.Estado.AT_Fase5
    End Function

    Shared Function AT_Fase5(ByVal quadro As Quadro) As Automato.Estado

        Dim auxValue As Double = Nothing
        Dim auxCell As Celula = Nothing

        For i As Integer = 0 To quadro.Matriz.GetUpperBound(0)
            For j As Integer = 0 To quadro.Matriz.GetUpperBound(1)
                auxCell = quadro.Matriz(i, j)
                If Not auxCell.FlagInferior And Not auxCell.FlagSuperior And auxCell IsNot quadro.PermElemento Then
                    auxCell.ValorInferior = Troca.getLineFlagElement(j, quadro) * Troca.getColFlagElement(i, quadro)
                End If
            Next
        Next

        Return Automato.Estado.AT_Fase6
    End Function

    Private Shared Function getLineFlagElement(ByVal col As Integer, ByVal quadro As Quadro) As Double

        Dim auxcell As Celula = Nothing

        For i As Integer = 0 To quadro.Matriz.GetUpperBound(0)
            If quadro.Matriz(i, col).FlagSuperior Then
                auxcell = quadro.Matriz(i, col)
                Exit For
            End If
        Next
        Return auxcell.ValorSuperior
    End Function

    Private Shared Function getColFlagElement(ByVal line As Integer, ByVal quadro As Quadro) As Double

        Dim auxcell As Celula = Nothing

        For i As Integer = 0 To quadro.Matriz.GetUpperBound(1)
            If quadro.Matriz(line, i).FlagInferior Then
                auxcell = quadro.Matriz(line, i)
                Exit For
            End If
        Next
        Return auxcell.ValorInferior

    End Function

    Shared Function AT_Fase6(ByVal quadro As Quadro) As Automato.Estado

        'Trocar cabeçalhos da linha e coluna permissiveis
        Dim auxHeader As String = String.Empty
        auxHeader = quadro.ColunaHeader(quadro.PermElementoPos.Y)
        quadro.ColunaHeader(quadro.PermElementoPos.Y) = quadro.LinhaHeader(quadro.PermElementoPos.X)
        quadro.LinhaHeader(quadro.PermElementoPos.X) = auxHeader

        'Passar itens da coluna e linha permissiveis para cima
        Dim auxcell As Celula = Nothing
        For i As Integer = 0 To quadro.Matriz.GetUpperBound(0)
            For j As Integer = 0 To quadro.Matriz.GetUpperBound(1)
                auxcell = quadro.Matriz(i, j)
                If i = quadro.PermElementoPos.X Or j = quadro.PermElementoPos.Y Then
                    auxcell.ValorSuperior = auxcell.ValorInferior
                Else
                    auxcell.ValorSuperior = auxcell.ValorInferior + auxcell.ValorSuperior
                End If
            Next
        Next

        Return Automato.Estado.PE_Fase1
    End Function

End Class
