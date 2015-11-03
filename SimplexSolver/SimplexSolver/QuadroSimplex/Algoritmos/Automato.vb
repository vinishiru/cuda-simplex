Public Class Automato

    Public Enum Estado
        'Primeira etapa
        PE_Fase1
        PE_Fase2
        PE_Fase3
        PE_Fase4
        'Segunda Etapa
        SE_Fase1
        SE_Fase2
        SE_Fase3
        'Troca
        AT_Fase1
        AT_Fase2
        AT_Fase3
        AT_Fase4
        AT_Fase5
        AT_Fase6
        'Finais
        NO_SOLUTION
        OPTIMAL_SOLUTION

    End Enum

    'Iniciar sempre da Primeira etapa fase 1
    Public Shared state As Estado = Estado.PE_Fase1

    Public Shared Sub reset()
        state = Estado.PE_Fase1
    End Sub

    ''' <summary>
    ''' Função central do algoritmo. Foi feita na forma de automato (máquina de estados) para permitir que se execute cada passo do algoritmo
    ''' separadamente.
    ''' Cada parte do algoritmo retorna um estado diferente. Por meio desse estado, sabe-se qual o próximo passo do algoritmo a ser executado.
    ''' </summary>
    ''' <param name="quadro"></param>
    ''' <param name="_step"></param>
    ''' <param name="tilSolve"></param>
    ''' <remarks></remarks>
    Public Shared Sub AutomatoSimplex(ByRef quadro As Quadro, ByVal _step As Boolean, ByVal tilSolve As Boolean)


        While _step Or tilSolve

            Select Case state
                Case Estado.PE_Fase1
                    state = PrimeiraEtapa.PE_Fase1(quadro)

                Case Estado.PE_Fase2
                    state = PrimeiraEtapa.PE_Fase2(quadro)

                Case Estado.PE_Fase3
                    state = PrimeiraEtapa.PE_Fase3(quadro)

                Case Estado.AT_Fase1
                    state = Troca.AT_Fase1(quadro)

                Case Estado.AT_Fase2
                    state = Troca.AT_Fase2(quadro)

                Case Estado.AT_Fase3
                    state = Troca.AT_Fase3(quadro)

                Case Estado.AT_Fase4
                    state = Troca.AT_Fase4(quadro)

                Case Estado.AT_Fase5
                    state = Troca.AT_Fase5(quadro)

                Case Estado.AT_Fase6
                    state = Troca.AT_Fase6(quadro)

                Case Estado.SE_Fase1
                    state = SegundaEtapa.SE_Fase1(quadro)

                Case Estado.SE_Fase2
                    state = SegundaEtapa.SE_Fase2(quadro)
                Case Estado.SE_Fase3
                    state = SegundaEtapa.SE_Fase3(quadro)

                Case Estado.NO_SOLUTION
                    tilSolve = False
                Case Estado.OPTIMAL_SOLUTION
                    tilSolve = False
            End Select


            If state = Estado.PE_Fase1 Then
                quadro.deFlagAll()
            End If

            _step = False
        End While

    End Sub

End Class
