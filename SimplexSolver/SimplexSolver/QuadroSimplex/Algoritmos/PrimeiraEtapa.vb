''' <summary>
''' Classe que representa todos os passos do Algoritmo de Primeira Etapa do Simplex.
''' </summary>
''' <remarks></remarks>
Public Class PrimeiraEtapa

    Shared Function PE_Fase1(ByVal quadro As Quadro) As Automato.Estado

        Dim auxCell As Celula = Nothing
        Dim exists As Boolean = False

        'Verificar na coluna das variaveis basicas e flagar
        For i As Integer = 1 To quadro.Matriz.GetUpperBound(0)
            auxCell = quadro.Matriz(i, 0)
            If auxCell.ValorSuperior < 0 Then
                auxCell.Flag = True
                exists = True
            End If
        Next

        If exists Then
            Return Automato.Estado.PE_Fase2
        Else
            Return Automato.Estado.SE_Fase1
        End If
    End Function

    Shared Function PE_Fase2(ByVal quadro As Quadro) As Automato.Estado


        Dim auxFlagCell As Celula = Nothing
        Dim auxCell As Celula = Nothing
        Dim exists As Boolean = False
        Dim colLimit As Integer = quadro.Matriz.GetUpperBound(1)

        For i As Integer = 0 To quadro.Matriz.GetUpperBound(0)
            auxFlagCell = quadro.Matriz(i, 0)

            If auxFlagCell.Flag Then
                For j As Integer = 0 To colLimit
                    auxCell = quadro.Matriz(i, j)

                    If Not auxCell.Flag And auxCell.ValorSuperior < 0 Then
                        auxCell.Flag = True
                        exists = True
                        colLimit = j

                    End If
                Next
            End If
        Next

        If exists Then
            Return Automato.Estado.PE_Fase3
        Else
            Return Automato.Estado.NO_SOLUTION
        End If
    End Function

    Shared Function PE_Fase3(ByVal quadro As Quadro) As Automato.Estado

        'Definir elemento permissivel
        Dim auxFlagCell As Celula = Nothing
        Dim auxCell As Celula = Nothing
        Dim exists As Boolean = False
        Dim value As Double = Double.MaxValue
        Dim x_index As Integer = 0
        Dim y_index As Integer = 0

        For i As Integer = 0 To quadro.Matriz.GetUpperBound(0)
            auxFlagCell = quadro.Matriz(i, 0)

            If auxFlagCell.Flag Then
                For j As Integer = 0 To quadro.Matriz.GetUpperBound(1)
                    auxCell = quadro.Matriz(i, j)

                    If auxCell.Flag And Not (auxCell Is auxFlagCell) Then
                        'Calcular valor
                        If value > auxCell.ValorSuperior / auxFlagCell.ValorSuperior And value <> auxCell.ValorSuperior / auxFlagCell.ValorSuperior Then
                            x_index = i
                            y_index = j
                            value = auxCell.ValorSuperior / auxFlagCell.ValorSuperior
                            Exit For
                        End If
                    End If
                Next
            End If
        Next
        'Marcar linha e coluna do elemento permissivel
        quadro.FlagPermElement(x_index, y_index)

        Return Automato.Estado.AT_Fase1

    End Function

End Class
