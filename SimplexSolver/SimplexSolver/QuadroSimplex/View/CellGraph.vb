''' <summary>
''' Classe responsável por desenhar o quadro simplex. Não será passada para C++ a princípio.
''' </summary>
''' <remarks></remarks>
Public Class CellGraph

    Public Shared Property Altura As Integer = 30
    Public Shared Property Largura As Integer = 60
    Public Shared Max_X As Integer = 0
    Public Shared Max_Y As Integer = 0
    Private Shared risco As New Pen(Brushes.Black, 2)
    Private Shared flagRisco As New Pen(Brushes.Red, 2)
    Public Shared fillBrush As SolidBrush = New SolidBrush(Color.Aquamarine)

    Public Shared Sub PaintCellGraphics(ByVal cell As Celula, ByRef g As Graphics, ByVal x As Integer, ByVal y As Integer)

        ' Offsets
        Dim offsetX As Integer = (x + 1) * Largura
        Dim offsetY As Integer = (y + 1) * Altura

        'Pontos
        Dim A As New Point With {.X = offsetX, .Y = offsetY}
        Dim B As New Point With {.X = Largura + offsetX, .Y = offsetY}
        Dim C As New Point With {.X = offsetX, .Y = Altura + offsetY}
        Dim D As New Point With {.X = Largura + offsetX, .Y = Altura + offsetY}

        'Semi-retangulo superior
        Dim sup As Point() = {A, B, C}

        'Semi-retangulo inferior
        Dim inf As Point() = {B, C, D}

        'Label superior
        Dim supValue As String = cell.ValorSuperior.ToString
        'Label inferior
        Dim infValue As String = cell.ValorInferior.ToString

        'Preencher poligonos, caso precise
        If cell.FlagSuperior Then
            g.FillPolygon(fillBrush, sup)
        ElseIf cell.FlagInferior Then
            g.FillPolygon(fillBrush, inf)
        End If

        'Plotar poligonos
        If cell.Flag Then
            g.DrawPolygon(flagRisco, sup)
            g.DrawPolygon(flagRisco, inf)
        Else
            g.DrawPolygon(risco, sup)
            g.DrawPolygon(risco, inf)
        End If


        'Plotar valores
        g.DrawString(supValue, New Font("Arial", 9), Brushes.Black, A)
        g.DrawString(infValue, New Font("Arial", 9), Brushes.Black, New Point(D.X - (Largura / 2), D.Y - (Altura / 2)))

        VerificarClipBounds(D.X, D.Y)
    End Sub

    Public Shared Sub PaintRowHeaderGraphics(ByVal header As String, ByRef g As Graphics, ByVal x As Integer, ByVal y As Integer)

        Dim risco As New Pen(Brushes.Black, 2)
        y += 1 'Ajustar coordenadas

        Dim offsetX As Integer = x * Largura
        Dim offsetY As Integer = y * Altura

        g.DrawRectangle(risco, New Rectangle(offsetX, offsetY, Largura, Altura))
        g.DrawString(header, New Font("Arial", 10, FontStyle.Bold), Brushes.Black, New Point(offsetX, offsetY))

        VerificarClipBounds(offsetX + Largura, offsetY + Altura)
    End Sub

    Public Shared Sub PaintColHeaderGraphics(ByVal header As String, ByRef g As Graphics, ByVal x As Integer, ByVal y As Integer)

        Dim risco As New Pen(Brushes.Black, 2)
        x += 1 'Ajustar coordenadas

        Dim offsetX As Integer = x * Largura
        Dim offsetY As Integer = y * Altura

        g.DrawRectangle(risco, New Rectangle(offsetX, offsetY, Largura, Altura))
        g.DrawString(header, New Font("Arial", 10, FontStyle.Bold), Brushes.Black, New Point(offsetX, offsetY))

        VerificarClipBounds(offsetX + Largura, offsetY + Altura)

    End Sub

    Private Shared Sub VerificarClipBounds(ByVal x As Single, ByVal y As Integer)

        If x > Max_X Then
            Max_X = x
        End If
        If y > Max_Y Then
            Max_Y = y
        End If

    End Sub
End Class
