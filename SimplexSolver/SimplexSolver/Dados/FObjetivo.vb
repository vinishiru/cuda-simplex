Public Class FObjetivo

    Public Enum Extremo
        Min
        Max
    End Enum

    'Propriedades que compõem uma função objetivo
    Property Variaveis As New Dictionary(Of String, Variavel) 'Conjunto de variaveis
    Property VariaveisArtificiais As New Dictionary(Of String, Variavel) 'Conjunto de variaveis artificiais.
    Property TermoLivre As Double = 0.0 'Armazena o termo livre da função
    Property Restricoes As New Dictionary(Of Integer, Restricao) 'Conjunto de restrições da função
    Property Extr As FObjetivo.Extremo = Extremo.Min 'Tipo de otimização a ser realizada.
    Property Normalizado As Boolean = False 'Flag para saber se a função está normalizada (forma padrão)

    ''' <summary>
    ''' Função que cria uma nova variável na função objetivo, dando seu nome baseando na quantidade de variáveis já existentes.
    ''' </summary>
    ''' <returns></returns>
    ''' <remarks></remarks>
    Function createNewVariavel() As Variavel

        Dim varName As String = "X" & Variaveis.Count + 1

        Dim varAux As New Variavel
        varAux.Nome = varName
        Variaveis.Add(varName, varAux)

        Return varAux
    End Function


    Sub deleteVariavel()
        Variaveis.Remove("X" & Variaveis.Count)
    End Sub

    ''' <summary>
    ''' Função que cria uma nova variável artificial, dando seu nome baseando na quantidade de variáveis da função objetivo e restrições
    ''' já existentes.
    ''' </summary>
    ''' <returns></returns>
    ''' <remarks></remarks>
    Function createNewVariavelArtificial() As Variavel

        Dim varName As String = "X" & Variaveis.Count + VariaveisArtificiais.Count + 1

        Dim varAux As New Variavel
        varAux.Nome = varName
        VariaveisArtificiais.Add(varName, varAux)
        Return varAux
    End Function

    Sub setVariavelValue(ByVal varName As String, ByVal varValue As Double)
        Dim varAux As Variavel = Variaveis(varName)
        varAux.Coeficiente = varValue
    End Sub

    Private Function getRestricao(ByVal index As Integer) As Restricao
        Dim restricaoAux As Restricao = Nothing

        'Verificar se ja existe restriçao nesse indice, se nao existir, criar nova restriçao
        If Not Restricoes.ContainsKey(index) Then
            restricaoAux = New Restricao
            Restricoes.Add(index, restricaoAux)
        Else
            restricaoAux = Restricoes(index)
        End If
        Return restricaoAux
    End Function

    Sub setRestricaoRelacionamentoValue(ByVal index As Integer, ByVal rel As Restricao.Relacionamento)

        Dim restricaoAux As Restricao = Me.getRestricao(index)
        restricaoAux.Relation = rel

    End Sub

    Sub setRestricaoVariavelValue(ByVal index As Integer, ByVal varName As String, ByVal varValue As Double)

        Dim restricaoAux As Restricao = Me.getRestricao(index)

        'Editar o valor alterado
        restricaoAux.setVariavelValue(varName, varValue)

    End Sub

    Sub setRestricaoTermoLivreValue(ByVal index As Integer, ByVal value As Double)
        Dim restricaoAux As Restricao = Me.getRestricao(index)
        restricaoAux.TermoLivre = value
    End Sub

    Public Overrides Function ToString() As String

        Dim foString As String = "F(x) = "

        If Not Normalizado Then
            For Each var As Variavel In Variaveis.Values

                If var.Coeficiente > 0 Then
                    foString += " + " & var.Coeficiente & var.Nome
                Else
                    foString += var.Coeficiente & var.Nome
                End If

            Next
        Else

            foString += "0 - ( "

            For Each var As Variavel In Variaveis.Values
                If var.Coeficiente > 0 Then
                    foString += " + " & var.Coeficiente & var.Nome
                Else
                    foString += var.Coeficiente & var.Nome
                End If
            Next

            foString += " )"
        End If

        Return foString
    End Function

    Sub normalizar()

        normalizarExtremo()

        If Not Me.Normalizado Then
            normalizarFuncaoObj()
        End If

        normalizarRestricoes()

    End Sub

    Private Sub normalizarExtremo()

        'Se for max
        If Me.Extr = Extremo.Max Then

            'Inverter sinal das variaveis da FO
            For Each var As Variavel In Variaveis.Values
                var.Coeficiente = var.Coeficiente * (-1)
            Next

            'Inverter relacionamento das restricoes
            For Each rest As Restricao In Restricoes.Values
                If rest.Relation = Restricao.Relacionamento.MaiorIgual Then
                    rest.Relation = Restricao.Relacionamento.MenorIgual
                Else
                    rest.Relation = Restricao.Relacionamento.MaiorIgual
                End If
            Next

        End If
    End Sub

    Private Sub normalizarFuncaoObj()

        For Each var As Variavel In Variaveis.Values
            var.Coeficiente = var.Coeficiente * (-1)
        Next

        Me.Normalizado = True

    End Sub

    Private Sub normalizarRestricoes()

        Dim auxVar As Variavel = Nothing

        For Each rest As Restricao In Restricoes.Values
            If Not rest.Normalizado Then

                auxVar = createNewVariavelArtificial()

                'Adicionar variaveis artificiais
                If rest.Relation = Restricao.Relacionamento.MenorIgual Then
                    'Se o relacionamento for menor

                    'Variavel artificial é positiva
                    auxVar.Coeficiente = 1
                    rest.VariavelArtificial = auxVar
                    rest.Normalizado = True

                Else
                    'Variavel artificial é negativa
                    auxVar.Coeficiente = -1
                    rest.VariavelArtificial = auxVar

                    'Inverter valores de termo livre
                    rest.TermoLivre = rest.TermoLivre * (-1)

                    'Inverter valores dos coeficientes das variaveis
                    For Each var As Variavel In rest.Variaveis.Values
                        var.Coeficiente = var.Coeficiente * (-1)
                    Next
                End If

                rest.Normalizado = True
            End If
        Next

    End Sub



End Class
