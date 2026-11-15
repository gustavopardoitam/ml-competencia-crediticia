"""Pruebas de la ingeniería de variables JAE (src/features/engineering_jae.py)."""

import numpy as np
import pandas as pd
import pytest

from features.engineering_jae import (
    AGE,
    DEBT_RATIO,
    DEPENDENTS,
    INCOME,
    LATE30,
    LATE60,
    LATE90,
    OPEN_LINES,
    REAL_ESTATE,
    UTIL,
    FeatureConfig,
    GroupMedianImputer,
    WoEEncoder,
    add_disposable_income,
    build_preprocessor,
    clean_raw,
    prepare_fixed,
    quantile_edges,
)


@pytest.fixture
def raw() -> pd.DataFrame:
    """Muestra cruda con los casos especiales detectados en el EDA."""
    return pd.DataFrame(
        {
            UTIL: [0.0, 0.5, 3.0, 50_000.0, 0.2, 0.9],
            AGE: [0, 30, 45, 60, 25, 70],
            LATE30: [0, 1, 98, 0, 2, 0],
            DEBT_RATIO: [0.3, 1500.0, 0.5, 0.2, 800.0, 0.4],
            INCOME: [5000.0, np.nan, 3000.0, 1_000_000.0, 0.0, 4000.0],
            OPEN_LINES: [1, 5, 8, 60, 3, 0],
            LATE90: [0, 0, 98, 0, 1, 0],
            REAL_ESTATE: [0, 1, 2, 54, 0, 1],
            LATE60: [0, 0, 98, 0, 0, 0],
            DEPENDENTS: [0.0, np.nan, 2.0, 13.0, 1.0, 0.0],
        }
    )


def test_clean_raw_convierte_codigos_96_98_en_faltantes(raw: pd.DataFrame) -> None:
    out = clean_raw(raw)
    assert out.loc[2, [LATE30, LATE60, LATE90]].isna().all()
    assert out["late_sentinel"].tolist() == [0, 0, 1, 0, 0, 0]


def test_clean_raw_invalida_edad_menor_a_18(raw: pd.DataFrame) -> None:
    out = clean_raw(raw)
    assert np.isnan(out.loc[0, AGE])
    assert out[AGE].notna().sum() == 5


def test_clean_raw_marca_indicadores_de_faltantes(raw: pd.DataFrame) -> None:
    out = clean_raw(raw)
    assert out["income_missing"].tolist() == [0, 1, 0, 0, 0, 0]
    assert out["income_zero"].tolist() == [0, 0, 0, 0, 1, 0]
    assert out["dependents_missing"].tolist() == [0, 1, 0, 0, 0, 0]


def test_clean_raw_falla_si_falta_una_columna(raw: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Faltan columnas"):
        clean_raw(raw.drop(columns=[INCOME]))


def test_clean_raw_no_modifica_la_entrada(raw: pd.DataFrame) -> None:
    original = raw.copy()
    clean_raw(raw)
    pd.testing.assert_frame_equal(raw, original)


def test_monto_de_deuda_depende_de_si_hay_ingreso(raw: pd.DataFrame) -> None:
    out = prepare_fixed(raw)
    # Con ingreso: razón × ingreso; sin ingreso o ingreso 0: DebtRatio tal cual
    assert out.loc[0, "debt_amount"] == pytest.approx(0.3 * 5000)
    assert out.loc[1, "debt_amount"] == pytest.approx(1500.0)
    assert out.loc[4, "debt_amount"] == pytest.approx(800.0)
    assert out.loc[1, "debt_ratio_valid"] == 0


def test_recortes_fijos(raw: pd.DataFrame) -> None:
    out = prepare_fixed(raw)
    assert out["util_clip"].max() == 2.0
    assert out["income_clip"].max() == 25_000.0
    assert out["open_lines"].max() == 24
    assert out["real_estate"].max() == 5
    assert out["dependents"].max() == 5


def test_indicadores_de_perfil(raw: pd.DataFrame) -> None:
    out = prepare_fixed(raw)
    assert out["util_zero"].tolist() == [1, 0, 0, 0, 0, 0]
    assert out["thin_file"].tolist() == [1, 0, 0, 0, 0, 0]
    assert out["util_gt1"].tolist() == [0, 0, 1, 1, 0, 0]


def test_edad_invalida_va_al_grupo_mas_joven(raw: pd.DataFrame) -> None:
    out = prepare_fixed(raw)
    assert out.loc[0, "age_group"] == 0
    assert out["age_group"].dtype.kind == "i"


def test_atrasos_con_codigo_quedan_en_cero_con_indicador(raw: pd.DataFrame) -> None:
    out = prepare_fixed(raw)
    assert out.loc[2, ["late_30_59", "late_60_89", "late_90"]].tolist() == [0, 0, 0]
    assert out.loc[2, "late_sentinel"] == 1


def test_quantile_edges_conserva_el_cero_en_variables_discretas() -> None:
    values = pd.Series([0] * 90 + [1] * 5 + [3] * 5)
    edges = quantile_edges(values, n_bins=10)
    bins = pd.cut(values, edges, labels=False)
    assert bins[values == 0].nunique() == 1
    assert bins[values == 0].iloc[0] != bins[values == 3].iloc[0]


def test_quantile_edges_valida_argumentos() -> None:
    with pytest.raises(ValueError):
        quantile_edges(pd.Series([1.0, 2.0]), n_bins=1)
    with pytest.raises(ValueError):
        quantile_edges(pd.Series([], dtype=float), n_bins=5)


def test_imputador_por_grupo_usa_la_mediana_del_grupo() -> None:
    df = pd.DataFrame(
        {"grupo": [1, 1, 1, 2, 2, 2], "x": [10.0, 20.0, np.nan, 100.0, 200.0, np.nan]}
    )
    out = GroupMedianImputer("x", "x_imp", group_cols=("grupo",)).fit_transform(df)
    assert out["x_imp"].tolist() == [10.0, 20.0, 15.0, 100.0, 200.0, 150.0]
    assert out["x"].isna().sum() == 2  # la columna original no se toca


def test_imputador_usa_mediana_global_en_grupos_no_vistos() -> None:
    ajuste = pd.DataFrame({"grupo": [1, 1, 2], "x": [10.0, 20.0, 30.0]})
    nuevo = pd.DataFrame({"grupo": [3], "x": [np.nan]})
    imputador = GroupMedianImputer("x", "x_imp", group_cols=("grupo",)).fit(ajuste)
    assert imputador.transform(nuevo)["x_imp"].iloc[0] == 20.0


def test_imputador_solo_aprende_de_los_datos_de_ajuste() -> None:
    ajuste = pd.DataFrame({"grupo": [1, 1], "x": [10.0, 20.0]})
    validacion = pd.DataFrame({"grupo": [1, 1], "x": [1000.0, np.nan]})
    imputador = GroupMedianImputer("x", "x_imp", group_cols=("grupo",)).fit(ajuste)
    assert imputador.transform(validacion)["x_imp"].tolist() == [1000.0, 15.0]


def test_imputador_positive_only_ignora_ceros() -> None:
    df = pd.DataFrame({"x": [0.0, 0.0, 0.0, 10.0, 30.0, np.nan]})
    out = GroupMedianImputer("x", "x_imp", positive_only=True).fit_transform(df)
    assert out["x_imp"].iloc[-1] == 20.0


def test_imputador_con_deciles_aprendidos() -> None:
    df = pd.DataFrame(
        {"deuda": np.arange(100, dtype=float), "x": np.arange(100, dtype=float)}
    )
    df.loc[95, "x"] = np.nan
    imputador = GroupMedianImputer("x", "x_imp", binned_col="deuda", n_bins=10)
    out = imputador.fit_transform(df)
    assert out["x_imp"].notna().all()
    assert 90 <= out.loc[95, "x_imp"] <= 99


def test_ingreso_disponible() -> None:
    df = pd.DataFrame({"ingreso": [5000.0, 1000.0], "debt_amount": [1500.0, 3000.0]})
    out = add_disposable_income(df, ["ingreso"])
    assert out["disposable_ingreso"].tolist() == [3500.0, -2000.0]


def test_woe_es_negativo_en_el_rango_de_mas_riesgo() -> None:
    x = pd.DataFrame({"v": [0.0] * 50 + [1.0] * 50})
    y = pd.Series([0] * 45 + [1] * 5 + [0] * 25 + [1] * 25)
    out = WoEEncoder(["v"], n_bins=2).fit_transform(x, y)
    assert out.loc[0, "woe_v"] > 0 > out.loc[99, "woe_v"]


def test_woe_da_rango_propio_a_los_faltantes() -> None:
    x = pd.DataFrame({"v": [1.0, 2.0, np.nan, np.nan] * 25})
    y = pd.Series([0, 0, 1, 1] * 25)
    encoder = WoEEncoder(["v"], n_bins=2).fit(x, y)
    out = encoder.transform(x)
    assert out.loc[2, "woe_v"] < 0
    assert encoder.iv_["v"] > 0


def test_woe_requiere_objetivo_binario() -> None:
    with pytest.raises(ValueError, match="binaria"):
        WoEEncoder(["v"]).fit(pd.DataFrame({"v": [1.0, 2.0]}), pd.Series([0, 2]))


@pytest.mark.parametrize(
    "config",
    [
        FeatureConfig(),
        FeatureConfig(income_strategy="AB", late_mode="binary"),
        FeatureConfig(income_strategy="global", scaler="robust"),
        FeatureConfig(representation="woe"),
        FeatureConfig(representation="splines", scaler="quantile"),
    ],
)
def test_preprocesador_lineal_no_deja_faltantes(
    raw: pd.DataFrame, config: FeatureConfig
) -> None:
    data = pd.concat([raw] * 5, ignore_index=True)
    y = pd.Series([0, 1] * (len(data) // 2))
    out = build_preprocessor(config).fit_transform(prepare_fixed(data), y)
    assert len(out) == len(data)
    assert out.isna().sum().sum() == 0


def test_preprocesador_de_arboles_sin_imputar_conserva_faltantes(
    raw: pd.DataFrame,
) -> None:
    config = FeatureConfig(for_trees=True, impute_for_trees=False)
    out = build_preprocessor(config).fit_transform(prepare_fixed(raw))
    assert out["income_clip"].isna().sum() == 1
    assert "income_imp_age_mortgage" not in out.columns
