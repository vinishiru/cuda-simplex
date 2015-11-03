
using System;
using System.Collections;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;


namespace SimplexSolver.CS.Dados
{

  public class Restricao
  {

   

    //Propriedades que compõem uma função de restrição
    public Dictionary<string, Variavel> Variaveis { get; set; }
    //Conjunto de variáveis
    public Relacionamento Relation { get; set; }
    //Um tipo de relacionamento
    public double TermoLivre { get; set; }
    //Valor do termo livre
    public Variavel VariavelArtificial { get; set; }
    //Variavel artificial que deve ser adicionada para tornar a restrição uma igualdade
    public bool Normalizado { get; set; }
    //Apenas uma variável de controle para saber se a função de restrição está normalizada (forma padrão)

    public void setVariavelValue(string varName, double varValue)
    {
      //Verificar se ja existe essa variavel na restriçao
      Variavel varAux = null;

      if (Variaveis.ContainsKey(varName))
      {
        varAux = Variaveis[varName];
      }
      else
      {
        varAux = new Variavel();
        varAux.Nome = varName;
        Variaveis.Add(varName, varAux);
      }

      varAux.Coeficiente = varValue;
    }

    public void deleteVariavel()
    {
      Variaveis.Remove("X" + Variaveis.Count);
    }

    public override string ToString()
    {

      string restString = string.Empty;


      if (!Normalizado)
      {
        foreach (Variavel var in this.Variaveis.Values)
        {
          if (var.Coeficiente > 0)
          {
            restString += " + " + var.Coeficiente + var.Nome;
          }
          else
          {
            restString += var.Coeficiente + var.Nome;
          }
        }

        if (this.Relation == Relacionamento.MaiorIgual)
        {
          restString += " >= ";
        }
        else
        {
          restString += " <= ";
        }

        restString += TermoLivre.ToString();

      }
      else
      {
        restString += VariavelArtificial.Nome + " = " + TermoLivre + " - (";

        foreach (Variavel var in this.Variaveis.Values)
        {
          if (var.Coeficiente > 0)
          {
            restString += " + " + var.Coeficiente + var.Nome;
          }
          else
          {
            restString += var.Coeficiente + var.Nome;
          }
        }

        restString += " ) ";
      }

      return restString;
    }

  }

}