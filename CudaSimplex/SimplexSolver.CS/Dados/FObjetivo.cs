using System;
using System.Collections;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;

namespace SimplexSolver.CS.Dados
{
  public class FObjetivo
  {
    //Propriedades que compõem uma função objetivo
    public Dictionary<string, Variavel> Variaveis { get; set; }
    //Conjunto de variaveis
    public Dictionary<string, Variavel> VariaveisBasicas { get; set; }
    //Conjunto de variaveis artificiais.
    public double TermoLivre { get; set; }
    //Armazena o termo livre da função
    public Dictionary<string, Restricao> Restricoes { get; set; }
    //Conjunto de restrições da função
    public Extremo Extremo { get; set; }
    //Tipo de otimização a ser realizada.
    public bool Normalizado { get; set; }
    //Flag para saber se a função está normalizada (forma padrão)

    public FObjetivo()
    {
      this.Extremo = Dados.Extremo.Minimizar;
      this.Variaveis = new Dictionary<string, Variavel>();
      this.VariaveisBasicas = new Dictionary<string, Variavel>();
      this.Restricoes = new Dictionary<string, Restricao>();
    }

    public Variavel AddVariavel()
    {

      string varName = "X" + Variaveis.Count + 1;

      Variavel varAux = new Variavel();
      varAux.Nome = varName;
      Variaveis.Add(varName, varAux);

      return varAux;
    }

    public Variavel AddVariavel(string nome, double coeficiente)
    {

      Variavel varAux = new Variavel
      {
        Nome = nome,
        Coeficiente = coeficiente
      };
      Variaveis.Add(nome, varAux);

      return varAux;
    }

    public Restricao AddRestricao()
    {
      string restName = string.Format("REST_{0}", this.Restricoes.Count.ToString());
      return AddRestricao(restName);
    }

    public Restricao AddRestricao(string restName)
    {
      Restricao rest = new Restricao();
      this.Restricoes.Add(restName, rest);
      return rest;
    }

    public void AddVariavelRestricao(string restName, string varName, double varValue)
    {
      Restricao restricaoAux = this.RecuperarRestricao(restName);
      restricaoAux.addVariavel(varName, varValue);
    }

    public void SetDesigualdadeRestricao(string restName, Desigualdade desigualdade)
    {
      Restricao restricaoAux = this.RecuperarRestricao(restName);
      restricaoAux.Desigualdade = desigualdade;
    }

    public void SetTermoLivreRestricao(string restName, double value)
    {
      Restricao restricaoAux = this.RecuperarRestricao(restName);
      restricaoAux.TermoLivre = value;
    }

    public override string ToString()
    {

      string foString = "F(x) = ";

      if (!Normalizado)
      {

        foreach (Variavel var in Variaveis.Values)
        {
          if (var.Coeficiente > 0)
          {
            foString += " + " + var.Coeficiente + var.Nome;
          }
          else
          {
            foString += var.Coeficiente + var.Nome;
          }

        }

      }
      else
      {
        foString += "0 - ( ";

        foreach (Variavel var in Variaveis.Values)
        {
          if (var.Coeficiente > 0)
          {
            foString += " + " + var.Coeficiente + var.Nome;
          }
          else
          {
            foString += var.Coeficiente + var.Nome;
          }
        }

        foString += " )";
      }

      return foString;
    }

    public Variavel CriarVariavelBasica()
    {

      string varName = string.Format("VB_{0}", Variaveis.Count + VariaveisBasicas.Count + 1);

      Variavel varAux = new Variavel();
      varAux.Nome = varName;
      VariaveisBasicas.Add(varName, varAux);
      return varAux;
    }

    public void normalizar()
    {
      normalizarExtremo();

      if (!this.Normalizado)
      {
        normalizarFuncaoObj();
      }

      normalizarRestricoes();

    }


    private Restricao RecuperarRestricao(string restName)
    {
      Restricao restricaoAux = null;

      //Verificar se ja existe restriçao nesse indice, se nao existir, criar nova restriçao
      if (!Restricoes.ContainsKey(restName))
      {
        restricaoAux = new Restricao();
        Restricoes.Add(restName, restricaoAux);
      }
      else
      {
        restricaoAux = Restricoes[restName];
      }
      return restricaoAux;
    }

    private void normalizarExtremo()
    {
      //Se for max

      if (this.Extremo == Extremo.Maximizar)
      {
        //Inverter sinal das variaveis da FO
        foreach (Variavel var in Variaveis.Values)
        {
          var.Coeficiente = var.Coeficiente * (-1);
        }

        //Inverter relacionamento das restricoes
        foreach (Restricao rest in Restricoes.Values)
        {
          if (rest.Desigualdade == Desigualdade.MaiorOuIgual)
          {
            rest.Desigualdade = Desigualdade.MenorOuIgual;
          }
          else
          {
            rest.Desigualdade = Desigualdade.MaiorOuIgual;
          }
        }

      }
    }

    private void normalizarFuncaoObj()
    {
      foreach (Variavel var in Variaveis.Values)
      {
        var.Coeficiente = var.Coeficiente * (-1);
      }

      this.Normalizado = true;

    }

    private void normalizarRestricoes()
    {
      Variavel auxVar = null;

      foreach (Restricao rest in Restricoes.Values)
      {

        if (!rest.Normalizado)
        {
          auxVar = CriarVariavelBasica();

          //Adicionar variaveis artificiais
          if (rest.Desigualdade == Desigualdade.MenorOuIgual)
          {
            //Se o relacionamento for menor

            //Variavel artificial é positiva
            auxVar.Coeficiente = 1;
            rest.VariavelArtificial = auxVar;
            rest.Normalizado = true;

          }
          else
          {
            //Variavel artificial é negativa
            auxVar.Coeficiente = -1;
            rest.VariavelArtificial = auxVar;

            //Inverter valores de termo livre
            rest.TermoLivre = rest.TermoLivre * (-1);

            //Inverter valores dos coeficientes das variaveis
            foreach (Variavel var in rest.Variaveis.Values)
            {
              var.Coeficiente = var.Coeficiente * (-1);
            }
          }

          rest.Normalizado = true;
        }
      }

    }
  }

}