''' <summary>
''' Classe que representa todos os passos do Algoritmo de Segunda Etapa do Simplex.
''' </summary>
''' <remarks></remarks>
Public Class SegundaEtapa

    Shared Function SE_Fase1(ByVal quadro As Quadro) As Automato.Estado

        Dim exists As Boolean = False
        Dim auxCell As Celula = Nothing
        'Procurar elemento positivo na linha F, sem contar com o termo livre!
        For col As Integer = 1 To quadro.Matriz.GetUpperBound(1)
            auxCell = quadro.Matriz(0, col)
            If auxCell.ValorSuperior > 0 Then
                auxCell.Flag = True
                exists = True
            End If
        Next

        If exists Then
            Return Automato.Estado.SE_Fase2
        Else
            Return Automato.Estado.OPTIMAL_SOLUTION
        End If
    End Function

    Shared Function SE_Fase2(ByVal quadro As Quadro) As Automato.Estado

        Dim exists As Boolean = False
        Dim auxFlagCell As Celula = Nothing
        Dim auxCell As Celula = Nothing
        Dim linLimit As Integer = quadro.Matriz.GetUpperBound(0)

        For i As Integer = 1 To quadro.Matriz.GetUpperBound(1)
            auxFlagCell = quadro.Matriz(0, i)
            If auxFlagCell.Flag Then
                For j As Integer = 0 To linLimit
                    auxCell = quadro.Matriz(i, j)

                    If Not auxCell.Flag And auxCell.ValorSuperior > 0 Then
                        auxCell.Flag = True
                        exists = True
                        linLimit = j
                        Exit For
                    End If
                Next
            End If
        Next

        If exists Then
            Return Automato.Estado.SE_Fase3
        Else
            Return Automato.Estado.NO_SOLUTION
        End If
    End Function

    Shared Function SE_Fase3(ByVal quadro As Quadro) As Automato.Estado

        Return PrimeiraEtapa.PE_Fase3(quadro)

    End Function

End Class
