# Customer Retention com Cell2Cell

## Objetivo

Este projeto estima a probabilidade de churn de clientes de telecomunicações e
gera uma fila priorizada para apoiar ações operacionais. O objetivo é comparar
um baseline de Regressão Logística com HistGradientBoosting, medir a qualidade
dos scores e ordenar clientes conforme o risco previsto.

## Base pública

Os dados são provenientes de uma fonte pública e incluem uma partição rotulada
para desenvolvimento e uma partição sem rótulo para pontuação.

**Cell2Cell Telecom Churn:** [fonte pública no Kaggle](https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom).

```python
ORIGEM_DADOS = "https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom"
```

A execução atual utilizou 51.047 clientes no conjunto rotulado e 20.000 no
holdout. A prevalência observada de churn no conjunto rotulado foi de 28,82%.

## Problema de churn

O alvo é `Churn`: `Yes` representa churn e `No` representa permanência. O
pipeline produz uma probabilidade de churn por cliente. Esse score descreve
associação estatística com o desfecho e não demonstra o efeito causal de uma
intervenção.

## Principais variáveis

O modelo usa 52 variáveis disponíveis no recorte de cliente, agrupadas em:

- uso e qualidade de chamadas, como `MonthlyMinutes`, `DroppedCalls` e
  `PercChangeMinutes`;
- cobrança e receita corrente, como `MonthlyRevenue` e
  `TotalRecurringCharge`;
- relacionamento e aparelho, como `MonthsInService`, `Handsets` e
  `CurrentEquipmentDays`;
- perfil e segmentação, como `ServiceArea`, `CreditRating`, `PrizmCode`,
  `Occupation` e `MaritalStatus`.

`CustomerID` não entra como feature. As variáveis `RetentionCalls`,
`RetentionOffersAccepted` e `MadeCallToRetentionTeam` são excluídas por
registrarem ações ou interações de retenção que podem ocorrer após o momento
de previsão. `AdjustmentsToCreditRating` permanece excluída até que seu
momento de disponibilidade seja confirmado.

## Preparação dos dados

Os CSVs são lidos diretamente do ZIP. Variáveis numéricas são convertidas para
número, valores ausentes recebem tratamento no pipeline e categorias ausentes
são marcadas como `AUSENTE`. As variáveis categóricas recebem codificação
adequada a cada algoritmo.

## Modelos e validação

- Regressão Logística como baseline, com padronização das variáveis numéricas e
  one-hot encoding das categorias.
- HistGradientBoosting como candidato principal, com codificação ordinal das
  categorias.

A base não possui data de observação. Por isso, a avaliação usa uma divisão
estratificada aleatória, com semente 42: 60% para treino, 20% para validação e
20% para teste final. A comparação e a escolha do modelo ocorrem somente na
validação. Depois, o modelo escolhido é reajustado com treino mais validação e
avaliado uma única vez no teste final. Não há validação temporal, backtest
temporal ou projeção de séries temporais.

## Métricas da execução atual

| Modelo | ROC AUC | Average Precision | Brier Score |
| --- | ---: | ---: | ---: |
| HistGradientBoosting | 0,6791 | 0,4560 | 0,1880 |
| Regressão Logística | 0,6141 | 0,3673 | 0,2403 |

As métricas acima são calculadas na validação e a seleção usa a maior Average
Precision. O HistGradientBoosting foi selecionado. No teste final, após o
reajuste com treino mais validação, obteve ROC AUC de 0,6745, Average Precision
de 0,4586 e Brier Score de 0,1885. As importâncias por permutação são
calculadas no teste final.

## Lift e priorização operacional

O ranking é ordenado pela probabilidade prevista de churn. No teste final, a
fila de 10% com maior risco teve precisão de 55,24%, lift de 1,92 e capturou
19,17% dos churns observados. As demais métricas de capacidade são calculadas
somente após a escolha do modelo na validação.

O arquivo `holdout_pontuado.csv` contém os 20.000 clientes sem rótulo,
ordenados por risco. Ele inclui posição, decil e uma exposição mensal por risco
calculada como `MonthlyRevenue × probabilidade_churn`. Essa exposição serve
para contextualizar a fila e não é uma projeção de receita futura ou impacto
financeiro causal.

## PSI entre amostras

O PSI foi mantido apenas para comparar o treino rotulado com o holdout sem
rótulo. Ele não é temporal. Na execução atual, o maior PSI foi 0,0021 para
`ServiceArea`, classificado como baixo.

## Como executar

1. Instale as dependências:

   ```powershell
   pip install -r requirements.txt
   ```

2. Aponte a variável de ambiente para o ZIP público da Cell2Cell:

   ```powershell
   $env:ARQUIVO_DADOS = "C:\caminho\para\cell2cell.zip"
   ```

3. Execute o pipeline:

   ```powershell
   python cell2cell_pipeline.py
   ```

Os arquivos são gerados em `outputs/tabelas`, `outputs/graficos` e
`outputs/modelos`.

## Limitações

- A base não oferece data de observação, portanto os resultados não avaliam
  generalização ao longo do tempo.
- O holdout não possui rótulo de churn, então serve para pontuação, não para
  cálculo de métricas.
- A disponibilidade exata das variáveis no momento da decisão deve ser
  confirmada antes de uso operacional.
- O modelo identifica padrões preditivos, sem estabelecer causalidade.
