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
    public Dictionary<string, Variavel> VariaveisArtificiais { get; set; }
    //Conjunto de variaveis artificiais.
    public double TermoLivre { get; set; }
    //Armazena o termo livre da função
    public Dictionary<int, Restricao> Restricoes { get; set; }
    //Conjunto de restrições da função
    public Extremo Extr { get; set; }
    //Tipo de otimização a ser realizada.
    public bool Normalizado { get; set; }
    //Flag para saber se a função está normalizada (forma padrão)

    /// <summary>
    /// Função que cria uma nova variável na função objetivo, dando seu nome baseando na quantidade de variáveis já existentes.
    /// </summary>
    /// <returns></returns>
    /// <remarks></remarks>
    public Variavel createNewVariavel()
    {

      string varName = "X" + Variaveis.Count + 1;

      Variavel varAux = new Variavel();
      varAux.Nome = varName;
      Variaveis.Add(varName, varAux);

      return varAux;
    }


    public void deleteVariavel()
    {
      Variaveis.Remove("X" + Variaveis.Count);
    }

    /// <summary>
    /// Função que cria uma nova variável artificial, dando seu nome baseando na quantidade de variáveis da função objetivo e restrições
    /// já existentes.
    /// </summary>
    /// <returns></returns>
    /// <remarks></remarks>
    public Variavel createNewVariavelArtificial()
    {

      string varName = "X" + Variaveis.Count + VariaveisArtificiais.Count + 1;

      Variavel varAux = new Variavel();
      varAux.Nome = varName;
      VariaveisArtificiais.Add(varName, varAux);
      return varAux;
    }

    public void setVariavelValue(string varName, double varValue)
    {
      Variavel varAux = Variaveis[varName];
      varAux.Coeficiente = varValue;
    }

    private Restricao getRestricao(int index)
    {
      Restricao restricaoAux = null;

      //Verificar se ja existe restriçao nesse indice, se nao existir, criar nova restriçao
      if (!Restricoes.ContainsKey(index))
      {
        restricaoAux = new Restricao();
        Restricoes.Add(index, restricaoAux);
      }
      else
      {
        restricaoAux = Restricoes[index];
      }
      return restricaoAux;
    }


    public void setRestricaoRelacionamentoValue(int index, Relacionamento rel)
    {
      Restricao restricaoAux = this.getRestricao(index);
      restricaoAux.Relation = rel;

    }


    public void setRestricaoVariavelValue(int index, string varName, double varValue)
    {
      Restricao restricaoAux = this.getRestricao(index);

      //Editar o valor alterado
      restricaoAux.setVariavelValue(varName, varValue);

    }

    public void setRestricaoTermoLivreValue(int index, double value)
    {
      Restricao restricaoAux = this.getRestricao(index);
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


    public void normalizar()
    {
      normalizarExtremo();

      if (!this.Normalizado)
      {
        normalizarFuncaoObj();
      }

      normalizarRestricoes();

    }


    private void normalizarExtremo()
    {
      //Se for max

      if (this.Extr == Extremo.Max)
      {
        //Inverter sinal das variaveis da FO
        foreach (Variavel var in Variaveis.Values)
        {
          var.Coeficiente = var.Coeficiente * (-1);
        }

        //Inverter relacionamento das restricoes
        foreach (Restricao rest in Restricoes.Values)
        {
          if (rest.Relation == Relacionamento.MaiorIgual)
          {
            rest.Relation = Relacionamento.MenorIgual;
          }
          else
          {
            rest.Relation = Relacionamento.MaiorIgual;
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
          auxVar = createNewVariavelArtificial();

          //Adicionar variaveis artificiais
          if (rest.Relation == Relacionamento.MenorIgual)
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