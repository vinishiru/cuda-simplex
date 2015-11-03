Imports SimplexSolver.Automato

''' <summary>
''' Classe que apenas retorna strings para a tela, mostrando em qual fase do processo o algoritmo se encontra.
''' </summary>
''' <remarks></remarks>
Public Class Status

    Public Property Algoritmo As String
    Public Property Fase As String
    Public Property Status As String

    Public Sub updateInfo(ByVal state As Automato.Estado)

        Select Case state
            Case Estado.PE_Fase1
                Algoritmo = "Algoritmo de Primeira Etapa"
                Fase = "Fase 1"
                Status = "Em andamento."

            Case Estado.PE_Fase2
                Algoritmo = "Algoritmo de Primeira Etapa"
                Fase = "Fase 2"
                Status = "Em andamento."

            Case Estado.PE_Fase3
                Algoritmo = "Algoritmo de Primeira Etapa"
                Fase = "Fase 3"
                Status = "Em andamento."

            Case Estado.AT_Fase1
                Algoritmo = "Algoritmo de Troca"
                Fase = "Fase 1"
                Status = "Em andamento."

            Case Estado.AT_Fase2
                Algoritmo = "Algoritmo de Troca"
                Fase = "Fase 2"
                Status = "Em andamento."

            Case Estado.AT_Fase3
                Algoritmo = "Algoritmo de Troca"
                Fase = "Fase 3"
                Status = "Em andamento."
            Case Estado.AT_Fase4
                Algoritmo = "Algoritmo de Troca"
                Fase = "Fase 4"
                Status = "Em andamento."

            Case Estado.AT_Fase5
                Algoritmo = "Algoritmo de Troca"
                Fase = "Fase 5"
                Status = "Em andamento."

            Case Estado.SE_Fase1
                Algoritmo = "Algoritmo de Segunda Etapa"
                Fase = "Fase 1"
                Status = "Em andamento."
            Case Estado.SE_Fase2
                Algoritmo = "Algoritmo de Segunda Etapa"
                Fase = "Fase 2"
                Status = "Em andamento."
            Case Estado.SE_Fase3
                Algoritmo = "Algoritmo de Segunda Etapa"
                Fase = "Fase 3"
                Status = "Em andamento."

            Case Estado.OPTIMAL_SOLUTION
                Status = "Problema tem solução ótima!"

            Case Estado.NO_SOLUTION
                Status = "Problema sem solução."

        End Select

    End Sub

End Class
