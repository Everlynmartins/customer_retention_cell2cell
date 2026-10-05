"""Treina e avalia modelos de churn com a base pública Cell2Cell.

Defina ARQUIVO_DADOS com o caminho para o ZIP da base. O pipeline lê os CSVs
diretamente do arquivo compactado e salva resultados novos em ``outputs``.
"""

import json
import os
from pathlib import Path
from zipfile import ZipFile

PASTA_PROJETO = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PASTA_PROJETO / ".matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


ORIGEM_DADOS = "https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom"
ARQUIVO_DADOS = Path(os.environ.get("ARQUIVO_DADOS", PASTA_PROJETO / "cell2cell.zip"))
SEMENTE = 42
PERCENTUAL_TESTE = 0.20
# Proporção da parcela de desenvolvimento reservada para validação.
PERCENTUAL_VALIDACAO = 0.25
PERCENTUAIS_CAPACIDADE = (0.05, 0.10, 0.20, 0.30)
MIN_CLIENTES_SEGMENTO = 100

PASTA_SAIDA = PASTA_PROJETO / "outputs"
PASTA_TABELAS = PASTA_SAIDA / "tabelas"
PASTA_GRAFICOS = PASTA_SAIDA / "graficos"
PASTA_MODELOS = PASTA_SAIDA / "modelos"
for pasta in (PASTA_TABELAS, PASTA_GRAFICOS, PASTA_MODELOS):
    pasta.mkdir(parents=True, exist_ok=True)

COLUNA_ID = "CustomerID"
COLUNA_ALVO = "Churn"
COLUNAS_VAZAMENTO = {
    "RetentionCalls": "Registra contato de retenção e pode ocorrer após a decisão de prever churn.",
    "RetentionOffersAccepted": "Registra resultado de uma oferta de retenção e pode ocorrer após a decisão de prever churn.",
    "MadeCallToRetentionTeam": "Registra interação com retenção e pode ocorrer após a decisão de prever churn.",
}
COLUNAS_REVISAR_MOMENTO = {
    "AdjustmentsToCreditRating": "Usar somente se a disponibilidade antes da previsão for confirmada.",
}

COLUNAS_NUMERICAS = [
    "MonthlyRevenue",
    "MonthlyMinutes",
    "TotalRecurringCharge",
    "DirectorAssistedCalls",
    "OverageMinutes",
    "RoamingCalls",
    "PercChangeMinutes",
    "PercChangeRevenues",
    "DroppedCalls",
    "BlockedCalls",
    "UnansweredCalls",
    "CustomerCareCalls",
    "ThreewayCalls",
    "ReceivedCalls",
    "OutboundCalls",
    "InboundCalls",
    "PeakCallsInOut",
    "OffPeakCallsInOut",
    "DroppedBlockedCalls",
    "CallForwardingCalls",
    "CallWaitingCalls",
    "MonthsInService",
    "UniqueSubs",
    "ActiveSubs",
    "Handsets",
    "HandsetModels",
    "CurrentEquipmentDays",
    "AgeHH1",
    "AgeHH2",
    "ReferralsMadeBySubscriber",
    "IncomeGroup",
]
COLUNAS_CATEGORICAS = [
    "ServiceArea",
    "ChildrenInHH",
    "HandsetRefurbished",
    "HandsetWebCapable",
    "TruckOwner",
    "RVOwner",
    "Homeownership",
    "BuysViaMailOrder",
    "RespondsToMailOffers",
    "OptOutMailings",
    "NonUSTravel",
    "OwnsComputer",
    "HasCreditCard",
    "NewCellphoneUser",
    "NotNewCellphoneUser",
    "OwnsMotorcycle",
    "HandsetPrice",
    "CreditRating",
    "PrizmCode",
    "Occupation",
    "MaritalStatus",
]
COLUNAS_MODELO = COLUNAS_NUMERICAS + COLUNAS_CATEGORICAS


def salvar_tabela(tabela, nome):
    caminho = PASTA_TABELAS / nome
    tabela.to_csv(caminho, index=False, encoding="utf_8_sig")
    return caminho


def ler_dados(arquivo_dados):
    """Lê as duas partições públicas sem extrair ou copiar os CSVs."""
    if not arquivo_dados.exists():
        raise FileNotFoundError(
            f"Arquivo de dados não encontrado: {arquivo_dados}. "
            "Defina a variável de ambiente ARQUIVO_DADOS com o caminho do ZIP."
        )
    if arquivo_dados.suffix.lower() != ".zip":
        raise ValueError("ARQUIVO_DADOS deve apontar para um arquivo ZIP da Cell2Cell.")

    with ZipFile(arquivo_dados) as arquivo_zip:
        nomes = set(arquivo_zip.namelist())
        esperados = {"cell2celltrain.csv", "cell2cellholdout.csv"}
        ausentes = esperados - nomes
        if ausentes:
            raise ValueError(f"O ZIP não contém os CSVs esperados: {sorted(ausentes)}")
        with arquivo_zip.open("cell2celltrain.csv") as arquivo_treino:
            treino = pd.read_csv(arquivo_treino)
        with arquivo_zip.open("cell2cellholdout.csv") as arquivo_holdout:
            holdout = pd.read_csv(arquivo_holdout)
    return treino, holdout


def auditar_vazamento(treino):
    linhas = [
        {
            "coluna": COLUNA_ID,
            "decisao": "excluir",
            "motivo": "Identificador único. Não descreve comportamento do cliente.",
        }
    ]
    linhas.extend(
        {"coluna": coluna, "decisao": "excluir", "motivo": motivo}
        for coluna, motivo in COLUNAS_VAZAMENTO.items()
    )
    linhas.extend(
        {"coluna": coluna, "decisao": "excluir até validação", "motivo": motivo}
        for coluna, motivo in COLUNAS_REVISAR_MOMENTO.items()
    )
    linhas.extend(
        {
            "coluna": coluna,
            "decisao": "usar",
            "motivo": "Variável disponível no recorte de cliente fornecido.",
        }
        for coluna in COLUNAS_MODELO
    )
    auditoria = pd.DataFrame(linhas).sort_values(["decisao", "coluna"])
    salvar_tabela(auditoria, "auditoria_vazamento.csv")

    esperadas = {
        COLUNA_ID,
        COLUNA_ALVO,
        *COLUNAS_VAZAMENTO,
        *COLUNAS_REVISAR_MOMENTO,
        *COLUNAS_MODELO,
    }
    faltantes = esperadas - set(treino.columns)
    if faltantes:
        raise ValueError(f"Colunas esperadas ausentes no treino: {sorted(faltantes)}")
    return auditoria


def preparar_dados(dados, exige_alvo):
    dados = dados.copy()
    if exige_alvo:
        valores_invalidos = set(dados[COLUNA_ALVO].dropna().unique()) - {"Yes", "No"}
        if valores_invalidos:
            raise ValueError(
                f"Valores inesperados em {COLUNA_ALVO}: {valores_invalidos}"
            )
        dados["churn"] = dados[COLUNA_ALVO].eq("Yes").astype("int8")

    for coluna in COLUNAS_NUMERICAS:
        dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")
    for coluna in COLUNAS_CATEGORICAS:
        dados[coluna] = dados[coluna].fillna("AUSENTE").astype("string")
    return dados


def gerar_qualidade_dados(treino, holdout):
    resumo = pd.DataFrame(
        {
            "coluna": treino.columns,
            "ausencia_treino_pct": treino.isna().mean().mul(100).round(3).to_numpy(),
            "ausencia_holdout_pct": holdout.isna().mean().mul(100).round(3).to_numpy(),
            "valores_unicos_treino": treino.nunique(dropna=False).to_numpy(),
        }
    )
    salvar_tabela(resumo, "qualidade_dados.csv")
    return resumo


def criar_modelos():
    transformador_logistico = ColumnTransformer(
        [
            (
                "numericas",
                Pipeline(
                    [
                        ("imputacao", SimpleImputer(strategy="median")),
                        ("escala", StandardScaler()),
                    ]
                ),
                COLUNAS_NUMERICAS,
            ),
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore", min_frequency=30),
                COLUNAS_CATEGORICAS,
            ),
        ]
    )
    transformador_arvores = ColumnTransformer(
        [
            (
                "categoricas",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                    min_frequency=30,
                    max_categories=100,
                ),
                COLUNAS_CATEGORICAS,
            ),
            ("numericas", "passthrough", COLUNAS_NUMERICAS),
        ]
    )
    return {
        "Regressão Logística": Pipeline(
            [
                ("preprocessamento", transformador_logistico),
                (
                    "classificador",
                    LogisticRegression(
                        C=1.0,
                        max_iter=1200,
                        class_weight="balanced",
                        random_state=SEMENTE,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            [
                ("preprocessamento", transformador_arvores),
                (
                    "classificador",
                    HistGradientBoostingClassifier(
                        learning_rate=0.07,
                        max_iter=180,
                        max_leaf_nodes=15,
                        min_samples_leaf=100,
                        l2_regularization=1.0,
                        random_state=SEMENTE,
                    ),
                ),
            ]
        ),
    }


def dividir_amostra(treino):
    separador_teste = StratifiedShuffleSplit(
        n_splits=1,
        test_size=PERCENTUAL_TESTE,
        random_state=SEMENTE,
    )
    indices_desenvolvimento, indices_teste = next(
        separador_teste.split(treino, treino["churn"])
    )
    desenvolvimento = treino.iloc[indices_desenvolvimento]
    separador_validacao = StratifiedShuffleSplit(
        n_splits=1,
        test_size=PERCENTUAL_VALIDACAO,
        random_state=SEMENTE,
    )
    indices_treino_relativos, indices_validacao_relativos = next(
        separador_validacao.split(desenvolvimento, desenvolvimento["churn"])
    )
    indices_treino = indices_desenvolvimento[indices_treino_relativos]
    indices_validacao = indices_desenvolvimento[indices_validacao_relativos]
    return indices_treino, indices_validacao, indices_teste


def metricas_ranking(alvo_observado, probabilidades, receita_mensal):
    ordem = np.argsort(probabilidades)[::-1]
    linhas = []
    taxa_base = alvo_observado.mean()
    for fracao in PERCENTUAIS_CAPACIDADE:
        quantidade = max(1, int(np.ceil(len(alvo_observado) * fracao)))
        selecionados = ordem[:quantidade]
        linhas.append(
            {
                "fracao_priorizada": fracao,
                "clientes_priorizados": quantidade,
                "precisao_observada": alvo_observado[selecionados].mean(),
                "lift": alvo_observado[selecionados].mean() / taxa_base,
                "captura_churn": alvo_observado[selecionados].sum()
                / alvo_observado.sum(),
                "receita_mensal_exposta": receita_mensal[selecionados].sum(),
                "receita_mensal_exposta_por_risco": (
                    receita_mensal[selecionados] * probabilidades[selecionados]
                ).sum(),
            }
        )
    return pd.DataFrame(linhas)


def calcular_metricas(nome, alvo_observado, probabilidades, nome_conjunto):
    return {
        "modelo": nome,
        "conjunto": nome_conjunto,
        "linhas": len(alvo_observado),
        "prevalencia_churn": alvo_observado.mean(),
        "roc_auc": roc_auc_score(alvo_observado, probabilidades),
        "average_precision": average_precision_score(alvo_observado, probabilidades),
        "brier_score": brier_score_loss(alvo_observado, probabilidades),
    }


def avaliar_modelo(nome, modelo, dados, indices_treino, indices_avaliacao, nome_conjunto):
    modelo.fit(
        dados.iloc[indices_treino][COLUNAS_MODELO],
        dados.iloc[indices_treino]["churn"],
    )
    alvo_observado = dados.iloc[indices_avaliacao]["churn"].to_numpy()
    probabilidades = modelo.predict_proba(dados.iloc[indices_avaliacao][COLUNAS_MODELO])[
        :, 1
    ]
    metricas = calcular_metricas(nome, alvo_observado, probabilidades, nome_conjunto)
    return metricas, probabilidades


def avaliar_modelos(treino):
    indices_treino, indices_validacao, indices_teste = dividir_amostra(treino)
    resultados = []
    for nome, modelo in criar_modelos().items():
        metricas, _ = avaliar_modelo(
            nome,
            modelo,
            treino,
            indices_treino,
            indices_validacao,
            "validacao",
        )
        resultados.append(metricas)

    tabela_metricas = pd.DataFrame(resultados).sort_values(
        ["average_precision", "roc_auc", "brier_score"],
        ascending=[False, False, True],
    )
    salvar_tabela(tabela_metricas, "metricas_validacao_modelos.csv")

    modelo_campeao = tabela_metricas.iloc[0]["modelo"]
    return (
        modelo_campeao,
        indices_treino,
        indices_validacao,
        indices_teste,
        tabela_metricas,
    )


def analisar_segmentos(treino):
    dimensoes = [
        "ServiceArea",
        "PrizmCode",
        "CreditRating",
        "IncomeGroup",
        "Occupation",
    ]
    linhas = []
    taxa_geral = treino["churn"].mean()
    for coluna in dimensoes:
        tabela = (
            treino.groupby(coluna, dropna=False, observed=True)
            .agg(
                clientes=(COLUNA_ID, "size"),
                churns=("churn", "sum"),
                taxa_churn=("churn", "mean"),
                receita_mensal=("MonthlyRevenue", "sum"),
            )
            .reset_index()
        )
        tabela = tabela.loc[tabela["clientes"] >= MIN_CLIENTES_SEGMENTO].copy()
        tabela["lift_churn"] = tabela["taxa_churn"] / taxa_geral
        tabela["dimensao"] = coluna
        tabela = tabela.rename(columns={coluna: "segmento"})
        linhas.append(tabela)
    segmentos = pd.concat(linhas, ignore_index=True).sort_values(
        ["lift_churn", "clientes"], ascending=[False, False]
    )
    salvar_tabela(segmentos, "segmentos_churn.csv")
    return segmentos


def calcular_psi(referencia, atual, coluna):
    limite = 1e-6
    if coluna in COLUNAS_NUMERICAS:
        valores_referencia = referencia[coluna].fillna(0)
        valores_atuais = atual[coluna].fillna(0)
        cortes = np.unique(np.quantile(valores_referencia, np.linspace(0, 1, 11)))
        if len(cortes) < 3:
            return 0.0
        cortes[0], cortes[-1] = -np.inf, np.inf
        distribuicao_referencia = pd.cut(
            valores_referencia, cortes, include_lowest=True
        ).value_counts(normalize=True, sort=False)
        distribuicao_atual = pd.cut(
            valores_atuais, cortes, include_lowest=True
        ).value_counts(normalize=True, sort=False)
    else:
        categorias = referencia[coluna].fillna("AUSENTE").astype(str)
        categorias_atuais = atual[coluna].fillna("AUSENTE").astype(str)
        principais = categorias.value_counts().head(20).index
        distribuicao_referencia = categorias.where(
            categorias.isin(principais), "OUTROS"
        ).value_counts(normalize=True)
        distribuicao_atual = categorias_atuais.where(
            categorias_atuais.isin(principais), "OUTROS"
        ).value_counts(normalize=True)
        categorias_psi = distribuicao_referencia.index.union(distribuicao_atual.index)
        distribuicao_referencia = distribuicao_referencia.reindex(
            categorias_psi, fill_value=0
        )
        distribuicao_atual = distribuicao_atual.reindex(categorias_psi, fill_value=0)

    distribuicao_referencia = np.clip(
        distribuicao_referencia.to_numpy(dtype=float), limite, None
    )
    distribuicao_atual = np.clip(distribuicao_atual.to_numpy(dtype=float), limite, None)
    return float(
        np.sum(
            (distribuicao_atual - distribuicao_referencia)
            * np.log(distribuicao_atual / distribuicao_referencia)
        )
    )


def avaliar_diferenca_amostras(treino, holdout):
    psi = pd.DataFrame(
        [
            {"variavel": coluna, "psi": calcular_psi(treino, holdout, coluna)}
            for coluna in COLUNAS_MODELO
        ]
    ).sort_values("psi", ascending=False)
    psi["interpretacao"] = pd.cut(
        psi["psi"],
        bins=[-np.inf, 0.10, 0.25, np.inf],
        labels=["baixo", "atencao", "alto"],
    )
    psi["comparacao"] = "treino rotulado versus holdout sem rótulo"
    salvar_tabela(psi, "psi_treino_vs_holdout.csv")
    return psi


def gerar_graficos(alvo_observado, probabilidades, nome_modelo):
    taxa_falsos_positivos, taxa_verdadeiros_positivos, _ = roc_curve(
        alvo_observado, probabilidades
    )
    precisao, recall, _ = precision_recall_curve(alvo_observado, probabilidades)
    fracao_positivos, probabilidade_media = calibration_curve(
        alvo_observado, probabilidades, n_bins=10, strategy="quantile"
    )
    calibracao = pd.DataFrame(
        {
            "probabilidade_media": probabilidade_media,
            "fracao_positivos": fracao_positivos,
        }
    )
    salvar_tabela(calibracao, "calibracao_modelo_campeao.csv")

    figura, eixos = plt.subplots(1, 3, figsize=(15, 4))
    eixos[0].plot(taxa_falsos_positivos, taxa_verdadeiros_positivos)
    eixos[0].plot([0, 1], [0, 1], color="gray", linestyle=":")
    eixos[0].set(
        title="Curva ROC",
        xlabel="Taxa de falso positivo",
        ylabel="Taxa de verdadeiro positivo",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    eixos[1].plot(recall, precisao)
    eixos[1].axhline(alvo_observado.mean(), color="gray", linestyle=":")
    eixos[1].set(
        title="Precisão e recall",
        xlabel="Recall",
        ylabel="Precisão",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    eixos[2].plot(probabilidade_media, fracao_positivos, marker="o")
    eixos[2].plot([0, 1], [0, 1], color="gray", linestyle=":")
    eixos[2].set(
        title="Calibração",
        xlabel="Probabilidade prevista",
        ylabel="Fração observada",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    figura.suptitle(f"Avaliação do modelo: {nome_modelo}")
    figura.tight_layout()
    figura.savefig(PASTA_GRAFICOS / "curvas_modelo_campeao.png", dpi=180)
    plt.close(figura)


def gerar_importancia(modelo, dados_teste):
    amostra = dados_teste.sample(min(12000, len(dados_teste)), random_state=SEMENTE)
    resultado = permutation_importance(
        modelo,
        amostra[COLUNAS_MODELO],
        amostra["churn"],
        scoring="average_precision",
        n_repeats=3,
        random_state=SEMENTE,
        n_jobs=1,
    )
    importancia = pd.DataFrame(
        {
            "variavel": COLUNAS_MODELO,
            "importancia_media": resultado.importances_mean,
            "desvio": resultado.importances_std,
        }
    ).sort_values("importancia_media", ascending=False)
    salvar_tabela(importancia, "importancia_permutacao.csv")
    return importancia


def pontuar_holdout(modelo, holdout):
    pontuado = holdout[[COLUNA_ID, "MonthlyRevenue", "TotalRecurringCharge"]].copy()
    pontuado["probabilidade_churn"] = modelo.predict_proba(holdout[COLUNAS_MODELO])[
        :, 1
    ]
    pontuado["receita_mensal_exposta_por_risco"] = (
        pontuado["MonthlyRevenue"].fillna(0) * pontuado["probabilidade_churn"]
    )
    pontuado = pontuado.sort_values("probabilidade_churn", ascending=False)
    pontuado["posicao_risco"] = np.arange(1, len(pontuado) + 1)
    pontuado["decil_risco"] = (
        pd.qcut(pontuado["probabilidade_churn"].rank(method="first"), 10, labels=False)
        + 1
    )
    salvar_tabela(pontuado, "holdout_pontuado.csv")
    return pontuado


def main():
    print(f"Lendo dados de: {ARQUIVO_DADOS}")
    treino_bruto, holdout_bruto = ler_dados(ARQUIVO_DADOS)
    auditar_vazamento(treino_bruto)
    gerar_qualidade_dados(treino_bruto, holdout_bruto)

    treino = preparar_dados(treino_bruto, exige_alvo=True)
    holdout = preparar_dados(holdout_bruto, exige_alvo=False)
    analisar_segmentos(treino)
    avaliar_diferenca_amostras(treino, holdout)

    (
        nome_campeao,
        indices_treino,
        indices_validacao,
        indices_teste,
        metricas_validacao,
    ) = avaliar_modelos(treino)
    print("\nMétricas de validação")
    print(metricas_validacao.round(4).to_string(index=False))

    modelo_campeao = criar_modelos()[nome_campeao]
    indices_desenvolvimento = np.concatenate([indices_treino, indices_validacao])
    modelo_campeao.fit(
        treino.iloc[indices_desenvolvimento][COLUNAS_MODELO],
        treino.iloc[indices_desenvolvimento]["churn"],
    )
    joblib.dump(modelo_campeao, PASTA_MODELOS / "modelo_churn.joblib")

    dados_teste = treino.iloc[indices_teste].copy()
    probabilidades = modelo_campeao.predict_proba(dados_teste[COLUNAS_MODELO])[:, 1]
    metricas_teste = pd.DataFrame(
        [
            calcular_metricas(
                nome_campeao,
                dados_teste["churn"].to_numpy(),
                probabilidades,
                "teste_final",
            )
        ]
    )
    ranking_teste = metricas_ranking(
        dados_teste["churn"].to_numpy(),
        probabilidades,
        dados_teste["MonthlyRevenue"].fillna(0).to_numpy(),
    )
    ranking_teste["modelo"] = nome_campeao
    ranking_teste["conjunto"] = "teste_final"
    salvar_tabela(metricas_teste, "metricas_teste_final.csv")
    salvar_tabela(ranking_teste, "politicas_capacidade.csv")
    print("\nMétricas do teste final")
    print(metricas_teste.round(4).to_string(index=False))

    gerar_graficos(dados_teste["churn"].to_numpy(), probabilidades, nome_campeao)
    gerar_importancia(modelo_campeao, dados_teste)
    pontuar_holdout(modelo_campeao, holdout)

    manifesto = {
        "origem_dados": ORIGEM_DADOS,
        "arquivo_dados": ARQUIVO_DADOS.name,
        "linhas_rotuladas": len(treino),
        "linhas_holdout": len(holdout),
        "prevalencia_churn": float(treino["churn"].mean()),
        "modelo_campeao": nome_campeao,
        "linhas_treino": len(indices_treino),
        "linhas_validacao": len(indices_validacao),
        "linhas_teste_final": len(indices_teste),
        "semente": SEMENTE,
        "validacao": "divisão estratificada aleatória em treino, validação e teste final; a base não possui data de observação.",
        "vazamento_excluido": sorted(COLUNAS_VAZAMENTO),
        "variaveis_revisao_momento": sorted(COLUNAS_REVISAR_MOMENTO),
    }
    (PASTA_SAIDA / "manifesto.json").write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
