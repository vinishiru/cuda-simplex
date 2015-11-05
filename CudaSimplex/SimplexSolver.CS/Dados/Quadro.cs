
using System;
using System.Collections;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;

namespace SimplexSolver.CS.Dados
{
  public class Quadro
  {

    public string[] ColunaHeader { get; set; }
    //Array que guarda o nome das variáveis do cabeçalho das colunas do quadro
    public string[] LinhaHeader { get; set; }
    //Array que guarda o nome das variáveis do cabeçalho das linhas do quadro
    public Celula[,] Matriz { get; set; }
    //Matriz que compoem o Quadro Simplex, este objeto é fundamental para o algoritmo.
    public Celula PermElemento { get; set; }
    //Armazena a celula do elemento permissivel encontrado no decorrer do algoritmo.
    //public Point PermElementoPos { get; set; }
    //Variavel para guardar a posição do elemento permissível quando encontrado.


    public Quadro(FObjetivo FuncaoObj)
    {

      Matriz = new Celula[FuncaoObj.VariaveisBasicas.Count + 1, FuncaoObj.Variaveis.Count + 1];
      ColunaHeader = new string[FuncaoObj.Variaveis.Count + 1];
      //+1 para o Bj
      LinhaHeader = new string[FuncaoObj.VariaveisBasicas.Count + 1];
      //+1 para funcao obj

      buildHeaders(FuncaoObj);

      buildMatriz(FuncaoObj);
    }


    private void buildHeaders(FObjetivo funcaoObj)
    {
      for (int i = 0; i <= ColunaHeader.Length - 1; i++)
      {
        if (i == 0)
        {
          ColunaHeader[i] = "Bj";
        }
        else
        {
          ColunaHeader[i] = funcaoObj.Variaveis["X" + i].Nome;
        }
      }

      for (int i = 0; i <= LinhaHeader.Length - 1; i++)
      {
        if (i == 0)
        {
          LinhaHeader[i] = "F";
        }
        else
        {
          LinhaHeader[i] = funcaoObj.VariaveisBasicas["X" + (i + funcaoObj.Variaveis.Count)].Nome;
        }
      }

    }


    private void buildMatriz(FObjetivo FuncaoObj)
    {
      //Linha da funcao objetivo
      Matriz[0, 0] = new Celula { ValorSuperior = FuncaoObj.TermoLivre };

      for (int i = 1; i <= FuncaoObj.Variaveis.Count; i++)
      {
        Matriz[0, i] = new Celula { ValorSuperior = FuncaoObj.Variaveis["X" + i].Coeficiente };
      }

      int linha = 1;

      foreach (Restricao rest in FuncaoObj.Restricoes.Values)
      {
        // Coluna 0 sera o termo livre
        Matriz[linha, 0] = new Celula { ValorSuperior = rest.TermoLivre };

        for (int col = 1; col <= FuncaoObj.Variaveis.Count; col++)
        {
          try
          {
            Matriz[linha, col] = new Celula { ValorSuperior = rest.Variaveis["X" + col].Coeficiente };
          }
          catch (Exception ex)
          {
            Matriz[linha, col] = new Celula { ValorSuperior = 0 };
          }
        }
        linha += 1;
      }

    }


    public int TotalLinhas()
    {
      return this.Matriz != null ? this.Matriz.GetLength(0) : 0;
    }

    public int TotalColunas()
    {
      return this.Matriz != null ? this.Matriz.GetLength(1) : 0;
    }

    //public Point PaintQuadroSimplex(Graphics g)
    //{

    //  g.Clear(Color.White);


    //  //Pintar Coluna Headers
    //  for (int i = 0; i <= ColunaHeader.Length - 1; i++)
    //  {
    //    CellGraph.PaintColHeaderGraphics(ColunaHeader(i), g, i, 0);
    //  }
    //  //Pintar Linha Headers
    //  for (int i = 0; i <= LinhaHeader.Length - 1; i++)
    //  {
    //    CellGraph.PaintRowHeaderGraphics(LinhaHeader(i), g, 0, i);
    //  }

    //  //Pintar celulas
    //  for (int i = 0; i <= Matriz.GetUpperBound(0); i++)
    //  {
    //    for (int j = 0; j <= Matriz.GetUpperBound(1); j++)
    //    {
    //      CellGraph.PaintCellGraphics(Matriz(i, j), g, j, i);
    //    }
    //  }

    //  return new Point(CellGraph.Max_X, CellGraph.Max_Y);
    //}


    //public void deFlagAll()
    //{
    //  for (int i = 0; i <= Matriz.GetUpperBound(0); i++)
    //  {
    //    for (int j = 0; j <= Matriz.GetUpperBound(1); j++)
    //    {
    //      Matriz[i, j].Flag = false;
    //      Matriz[i, j].FlagInferior = false;
    //      Matriz[i, j].FlagSuperior = false;
    //      Matriz[i, j].ValorInferior = 0.0;
    //    }
    //  }

    //}


    //public void FlagPermElement(int x_index, int y_index)
    //{
    //  //Desmarcar todas
    //  this.deFlagAll();

    //  for (int i = 0; i <= Matriz.GetUpperBound(0); i++)
    //  {
    //    for (int j = 0; j <= Matriz.GetUpperBound(1); j++)
    //    {
    //      if (i == x_index | j == y_index)
    //      {
    //        Matriz[i, j].Flag = true;
    //      }
    //    }
    //  }
    //  //Definir elemento permissivel
    //  PermElemento = Matriz[x_index, y_index];
    //  PermElementoPos = new Point(x_index, y_index);
    //}

  }
}