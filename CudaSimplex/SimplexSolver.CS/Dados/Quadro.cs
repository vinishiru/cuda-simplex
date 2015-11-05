
using System;
using System.Collections;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Linq;
using System.Text;

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
          LinhaHeader[i] = funcaoObj.VariaveisBasicas.ToList()[i - 1].Key;
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

    public override string ToString()
    {

      StringBuilder builder = new StringBuilder();

      //escrever os cabecalhos inicialmente
      builder.Append(string.Concat("\t", string.Join("\t", this.ColunaHeader)));

      builder.AppendLine();

      //escrever cabecalho da linha, seguido dos valores
      for (int i = 0; i < this.TotalLinhas(); i++)
      {
        //imprimir cabecalho da linha
        builder.Append(string.Concat(this.LinhaHeader[i], "\t"));
        for (int j = 0; j < this.TotalColunas(); j++)
        {
          builder.Append(string.Concat(this.Matriz[i, j].ValorSuperior, "\t"));
        }

        //fim da linha
        builder.AppendLine();
      }

      return builder.ToString();
    }

  }
}