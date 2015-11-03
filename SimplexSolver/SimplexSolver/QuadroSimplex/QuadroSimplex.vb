Public Class QuadroSimplex

    Dim quadro As Quadro
    Dim status As Status


    Public Sub New(ByVal fobj As FObjetivo)

        ' This call is required by the designer.
        InitializeComponent()

        ' Add any initialization after the InitializeComponent() call.
        quadro = New Quadro(fobj)
        status = New Status

    End Sub

    Private Sub canvas_Paint(ByVal sender As Object, ByVal e As System.Windows.Forms.PaintEventArgs) Handles canvas.Paint

        Dim visPoint As Point = quadro.PaintQuadroSimplex(e.Graphics)

        If Not e.Graphics.IsVisible(visPoint) Then
            'Reajustar size do canvas
            canvas.Size = New Size(visPoint.X + 10, visPoint.Y + 10)
            container.AutoScrollMinSize = New Size(visPoint.X + 10, visPoint.Y + 10)
        End If

    End Sub

    Private Sub btnStep_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles btnStep.Click

        'Avancar um step no automato
        Automato.AutomatoSimplex(quadro, True, False)

        'Repintar canvas do quadro
        canvas.Invalidate()
        updateInfoLabels()
    End Sub

    Private Sub QuadroSimplex_FormClosed(ByVal sender As Object, ByVal e As System.Windows.Forms.FormClosedEventArgs) Handles Me.FormClosed
        Automato.reset()
    End Sub

    Private Sub updateInfoLabels()
        status.updateInfo(Automato.state)
        lblAlg.Text = status.Algoritmo
        lblFase.Text = status.Fase
        lblStatus.Text = status.Status
    End Sub
    Private Sub QuadroSimplex_Load(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles MyBase.Load
        updateInfoLabels()
    End Sub

    Private Sub btnSolve_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles btnSolve.Click

        'Avancar um step no automato
        Automato.AutomatoSimplex(quadro, False, True)

        'Repintar canvas do quadro
        canvas.Invalidate()
        updateInfoLabels()

    End Sub
End Class