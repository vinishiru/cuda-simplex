Public Class Restricao

    Public Enum Relacionamento
        MenorIgual = 0
        MaiorIgual = 1
    End Enum

    'Propriedades que compõem uma função de restrição
    Property Variaveis As New Dictionary(Of String, Variavel) 'Conjunto de variáveis
    Property Relation As Relacionamento = Nothing 'Um tipo de relacionamento
    Property TermoLivre As Double 'Valor do termo livre
    Property VariavelArtificial As Variavel 'Variavel artificial que deve ser adicionada para tornar a restrição uma igualdade
    Property Normalizado As Boolean = False 'Apenas uma variável de controle para saber se a função de restrição está normalizada (forma padrão)


    Sub setVariavelValue(ByVal varName As String, ByVal varValue As Double)

        'Verificar se ja existe essa variavel na restriçao
        Dim varAux As Variavel = Nothing

        If Variaveis.ContainsKey(varName) Then
            varAux = Variaveis(varName)
        Else
            varAux = New Variavel
            varAux.Nome = varName
            Variaveis.Add(varName, varAux)
        End If

        varAux.Coeficiente = varValue
    End Sub

    Sub deleteVariavel()
        Variaveis.Remove("X" & Variaveis.Count)
    End Sub

    Public Overrides Function ToString() As String

        Dim restString As String = String.Empty

        If Not Normalizado Then

            For Each var As Variavel In Me.Variaveis.Values
                If var.Coeficiente > 0 Then
                    restString += " + " & var.Coeficiente & var.Nome
                Else
                    restString += var.Coeficiente & var.Nome
                End If
            Next

            If Me.Relation = Relacionamento.MaiorIgual Then
                restString += " >= "
            Else
                restString += " <= "
            End If

            restString += TermoLivre.ToString
        Else

            restString += VariavelArtificial.Nome & " = " & TermoLivre & " - ("

            For Each var As Variavel In Me.Variaveis.Values
                If var.Coeficiente > 0 Then
                    restString += " + " & var.Coeficiente & var.Nome
                Else
                    restString += var.Coeficiente & var.Nome
                End If
            Next

            restString += " ) "
        End If

        Return restString
    End Function

End Class
