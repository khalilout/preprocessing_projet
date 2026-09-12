import os
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

API_URL = os.environ.get("API_URL", "https://preprocessing-projet.onrender.com")

st.set_page_config(
    page_title="Prétraitement des données",
    page_icon="🧹",
    layout="wide",
)

st.title("🧹 Application de prétraitement des données")
st.caption("Nettoyage, imputation, gestion des outliers, scaling et encodage")

if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None
if "analysis" not in st.session_state:
    st.session_state.analysis = None

st.header("1. Analyse initiale des données")

uploaded_file = st.file_uploader("Charger un fichier CSV ou Excel", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    if st.button("📤 Charger et analyser", type="primary"):
        with st.spinner("Analyse en cours..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(f"{API_URL}/upload", files=files)

        if response.status_code == 200:
            result = response.json()
            st.session_state.dataset_id = result["dataset_id"]
            st.session_state.analysis = result
            st.success(f"Fichier chargé avec succès (dataset_id: {result['dataset_id'][:8]}...)")
        else:
            st.error(f"Erreur : {response.json().get('detail', response.text)}")

if st.session_state.analysis:
    analysis = st.session_state.analysis
    dataset_id = st.session_state.dataset_id

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lignes", analysis["n_rows"])
    col2.metric("Colonnes", analysis["n_columns"])
    col3.metric("Lignes dupliquées", analysis["duplicated_rows"])
    col4.metric("Mémoire (Mo)", analysis["memory_usage_mb"])

    if analysis["is_time_series"]:
        st.info(f"⏱️ Données de type **time series** détectées sur : {', '.join(analysis['time_series_columns'])}")

    st.subheader("Types de colonnes & valeurs manquantes")
    columns_df = pd.DataFrame(analysis["columns"])
    columns_df["missing_rate_%"] = (columns_df["missing_rate"] * 100).round(2)
    st.dataframe(
        columns_df[["name", "inferred_type", "dtype", "missing_count", "missing_rate_%", "n_unique", "skewness"]],
        use_container_width=True,
    )

    st.subheader("Taux de valeurs manquantes par colonne")
    fig_missing = px.bar(
        columns_df.sort_values("missing_rate_%", ascending=False),
        x="name", y="missing_rate_%",
        labels={"name": "Colonne", "missing_rate_%": "% de valeurs manquantes"},
        color="missing_rate_%",
        color_continuous_scale="Reds",
    )
    fig_missing.add_hline(y=5, line_dash="dash", line_color="gray",
                           annotation_text="Seuil 5% (méthodes avancées au-delà)")
    st.plotly_chart(fig_missing, use_container_width=True)

    numeric_cols = columns_df[columns_df["skewness"].notna()]
    if not numeric_cols.empty:
        st.subheader("Symétrie (skewness) des colonnes numériques")
        fig_skew = px.bar(
            numeric_cols.sort_values("skewness"),
            x="name", y="skewness",
            labels={"name": "Colonne", "skewness": "Skewness"},
            color="skewness",
            color_continuous_scale="RdBu",
        )
        fig_skew.add_hline(y=0.5, line_dash="dash", line_color="orange")
        fig_skew.add_hline(y=-0.5, line_dash="dash", line_color="orange")
        st.plotly_chart(fig_skew, use_container_width=True)
        st.caption("Au-delà de |0,5| : distribution considérée comme asymétrique (impacte le choix d'imputation et de scaling).")

    if analysis["correlation_matrix"]:
        st.subheader("Matrice de corrélation")
        corr_df = pd.DataFrame(analysis["correlation_matrix"])
        fig_corr = px.imshow(
            corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            aspect="auto",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("Aperçu des données (10 premières lignes)"):
        preview_resp = requests.get(f"{API_URL}/datasets/{dataset_id}/preview")
        if preview_resp.status_code == 200:
            st.dataframe(pd.DataFrame(preview_resp.json()), use_container_width=True)
        else:
            st.error(f"Impossible de charger l'aperçu : {preview_resp.status_code} — {preview_resp.text[:300]}")
            st.caption("Si l'erreur mentionne 'dataset_id introuvable', le backend a probablement redémarré (crash mémoire ?). Réuploadez le fichier.")

    st.divider()
    st.header("2. Traitement des valeurs manquantes")

    with st.expander("📊 Visualiser le motif des valeurs manquantes (matrice missingno)"):
        matrix_response = requests.get(f"{API_URL}/datasets/{dataset_id}/missing-matrix")
        if matrix_response.status_code == 200:
            st.image(matrix_response.content, use_column_width=True)
            st.caption("Chaque ligne blanche = une valeur manquante. Un motif structuré suggère un lien avec d'autres colonnes (MAR) ; un motif dispersé sans structure suggère du hasard pur (MCAR).")

    if st.button("🔍 Obtenir les recommandations d'imputation"):
        with st.spinner("Calcul des recommandations (tests statistiques MCAR/MAR)..."):
            strat_response = requests.get(f"{API_URL}/datasets/{dataset_id}/missing-strategy")
        if strat_response.status_code == 200:
            st.session_state.recommendations = strat_response.json()["recommendations"]
        else:
            st.error(f"Erreur : {strat_response.text}")

    if st.session_state.get("recommendations"):
        recos = st.session_state.recommendations

        if not recos:
            st.success("Aucune valeur manquante détectée — rien à traiter à cette étape ✅")
        else:
            st.write("Méthode recommandée par colonne, déterminée par test statistique (p-value) MCAR/MAR :")

            method_options = [
                "ne_pas_imputer", "moyenne", "mediane", "mode", "interpolation_lineaire",
                "bfill", "ffill", "knn_imputer", "iterative_imputer",
                "groupe_temporel", "mode_par_groupe", "suppression",
            ]

            chosen_strategies = {}
            for reco in recos:
                col_left, col_right = st.columns([3, 2])
                with col_left:
                    badge = "🟢 MCAR" if reco["missingness_type"] == "MCAR" else ("🟠 MAR/MNAR" if reco["missingness_type"] != "N/A" else "")
                    pval_txt = f" (p={reco['p_value']})" if reco["p_value"] is not None else ""
                    st.markdown(f"**{reco['column']}** — {reco['missing_rate']*100:.1f}% manquants {badge}{pval_txt}")
                    st.caption(reco["rationale"])
                with col_right:
                    default_index = (
                        method_options.index(reco["recommended_method"])
                        if reco["recommended_method"] in method_options else 0
                    )
                    chosen_strategies[reco["column"]] = st.selectbox(
                        "Méthode", method_options, index=default_index,
                        key=f"method_{reco['column']}", label_visibility="collapsed",
                    )

            if st.button("✅ Appliquer le traitement", type="primary"):
                with st.spinner("Imputation en cours..."):
                    impute_response = requests.post(
                        f"{API_URL}/datasets/{dataset_id}/impute-missing",
                        json={"strategies": chosen_strategies},
                    )
                if impute_response.status_code == 200:
                    result = impute_response.json()
                    st.session_state.analysis = result["analysis"]
                    st.session_state.recommendations = None
                    st.success("Traitement appliqué avec succès !")
                    st.write("**Journal des opérations :**")
                    st.dataframe(pd.DataFrame(result["log"]), use_container_width=True)
                    st.info("Les métriques ci-dessus (section 1) reflètent maintenant le dataset traité.")
                else:
                    st.error(f"Erreur : {impute_response.text}")

    st.divider()
    st.header("3. Traitement des valeurs aberrantes (outliers)")

    if st.button("🔍 Analyser les outliers"):
        with st.spinner("Analyse en cours..."):
            r = requests.get(f"{API_URL}/datasets/{dataset_id}/outlier-strategy")
        if r.status_code == 200:
            st.session_state.outlier_recos = r.json()["recommendations"]
            st.session_state.outlier_before_images = {}
        else:
            st.error(f"Erreur : {r.text}")

    if st.session_state.get("outlier_recos"):
        recos = st.session_state.outlier_recos
        if not recos:
            st.success("Aucune colonne numérique à traiter ✅")
        else:
            method_options = ["ne_pas_traiter", "zscore", "iqr", "winsorisation"]
            chosen = {}

            st.write("**Distribution avant traitement** (boxplot + histogramme) :")
            for reco in recos:
                col = reco["column"]
                c1, c2 = st.columns([3, 2])
                with c1:
                    st.markdown(f"**{col}** — skew={reco['skewness']}, {reco['n_outliers_detected']} outliers détectés")
                    st.caption(reco["rationale"])
                    if col not in st.session_state.outlier_before_images:
                        img = requests.get(
                            f"{API_URL}/datasets/{dataset_id}/outlier-distribution",
                            params={"column": col, "label": "AVANT", "color": "#f7a3a3"},
                        )
                        if img.status_code == 200:
                            st.session_state.outlier_before_images[col] = img.content
                    if col in st.session_state.outlier_before_images:
                        st.image(st.session_state.outlier_before_images[col], use_column_width=True)
                with c2:
                    idx = method_options.index(reco["recommended_method"]) if reco["recommended_method"] in method_options else 0
                    chosen[col] = st.selectbox(
                        "Méthode", method_options, index=idx,
                        key=f"outlier_{col}", label_visibility="collapsed",
                    )

            if st.button("✅ Traiter les outliers", type="primary"):
                with st.spinner("Traitement en cours..."):
                    r = requests.post(f"{API_URL}/datasets/{dataset_id}/treat-outliers", json={"strategies": chosen})
                if r.status_code == 200:
                    result = r.json()
                    st.session_state.analysis = result["analysis"]
                    st.success("Outliers traités !")
                    st.dataframe(pd.DataFrame(result["log"]), use_container_width=True)

                    st.write("**Comparaison de la distribution avant / après :**")
                    for col in chosen:
                        if chosen[col] == "ne_pas_traiter":
                            continue
                        img_after = requests.get(
                            f"{API_URL}/datasets/{dataset_id}/outlier-distribution",
                            params={"column": col, "label": "APRÈS", "color": "#a3f7b0"},
                        )
                        if col in st.session_state.outlier_before_images and img_after.status_code == 200:
                            st.markdown(f"**{col}**")
                            c1, c2 = st.columns(2)
                            c1.image(st.session_state.outlier_before_images[col], caption="Avant", use_column_width=True)
                            c2.image(img_after.content, caption="Après", use_column_width=True)

                    st.session_state.outlier_recos = None
                    st.session_state.outlier_before_images = {}
                else:
                    st.error(f"Erreur : {r.text}")

    st.divider()
    st.header("4. Feature scaling (mise à l'échelle)")

    if st.button("🔍 Analyser la distribution des colonnes"):
        with st.spinner("Analyse en cours..."):
            r = requests.get(f"{API_URL}/datasets/{dataset_id}/scaling-strategy")
        if r.status_code == 200:
            st.session_state.scaling_recos = r.json()["recommendations"]
        else:
            st.error(f"Erreur : {r.text}")

    if st.session_state.get("scaling_recos"):
        recos = st.session_state.scaling_recos
        if not recos:
            st.success("Aucune colonne numérique à mettre à l'échelle ✅")
        else:
            method_options = ["ne_pas_scaler", "minmax", "standard", "robust"]
            chosen = {}
            for reco in recos:
                c1, c2 = st.columns([3, 2])
                with c1:
                    st.markdown(f"**{reco['column']}** — skew={reco['skewness']}")
                    st.caption(reco["rationale"])
                with c2:
                    idx = method_options.index(reco["recommended_method"])
                    chosen[reco["column"]] = st.selectbox(
                        "Scaler", method_options, index=idx,
                        key=f"scaling_{reco['column']}", label_visibility="collapsed",
                    )
            if st.button("✅ Appliquer le scaling", type="primary"):
                with st.spinner("Scaling en cours..."):
                    r = requests.post(f"{API_URL}/datasets/{dataset_id}/apply-scaling", json={"strategies": chosen})
                if r.status_code == 200:
                    result = r.json()
                    st.session_state.analysis = result["analysis"]
                    st.session_state.scaling_recos = None
                    st.success("Scaling appliqué !")
                    st.dataframe(pd.DataFrame(result["log"]), use_container_width=True)
                else:
                    st.error(f"Erreur : {r.text}")

    st.divider()
    st.header("5. Encodage des variables catégorielles")

    if st.button("🔍 Analyser les colonnes catégorielles"):
        with st.spinner("Analyse en cours..."):
            r = requests.get(f"{API_URL}/datasets/{dataset_id}/encoding-strategy")
        if r.status_code == 200:
            st.session_state.encoding_recos = r.json()["recommendations"]
        else:
            st.error(f"Erreur : {r.text}")

    if st.session_state.get("encoding_recos"):
        recos = st.session_state.encoding_recos
        if not recos:
            st.success("Aucune colonne catégorielle à encoder ✅")
        else:
            encoding_choices = {}
            ordinal_orders = {}
            method_options = ["ne_pas_encoder", "label", "one_hot", "ordinal"]

            for reco in recos:
                col = reco["column"]
                c1, c2 = st.columns([3, 2])
                with c1:
                    st.markdown(f"**{col}** — {reco['n_unique']} catégories")
                    st.caption(reco["rationale"])
                with c2:
                    idx = method_options.index(reco["recommended_method"]) if reco["recommended_method"] in method_options else 0
                    encoding_choices[col] = st.selectbox(
                        "Méthode", method_options, index=idx,
                        key=f"encoding_method_{col}", label_visibility="collapsed",
                    )

                if encoding_choices[col] == "ordinal":
                    unique_values = reco.get("unique_values") or []
                    if not unique_values:
                        st.error(f"Valeurs uniques indisponibles pour {col}.")
                        continue
                    st.caption(f"Ordre pour **{col}** (sélectionne du plus petit au plus grand) :")
                    selected_order = st.multiselect(
                        f"Ordre des catégories — {col}",
                        options=unique_values,
                        default=[],
                        key=f"ordinal_order_{col}",
                        label_visibility="collapsed",
                    )
                    if len(selected_order) == len(unique_values):
                        ordinal_orders[col] = selected_order
                    else:
                        st.warning(
                            f"⚠️ {col} : sélectionne les {len(unique_values)} catégories "
                            f"({len(selected_order)}/{len(unique_values)})."
                        )

            if st.button("✅ Appliquer l'encodage", type="primary"):
                missing_orders = [
                    c for c, m in encoding_choices.items()
                    if m == "ordinal" and c not in ordinal_orders
                ]
                if missing_orders:
                    st.error(f"Ordre incomplet pour : {', '.join(missing_orders)}.")
                else:
                    with st.spinner("Encodage en cours..."):
                        r = requests.post(
                            f"{API_URL}/datasets/{dataset_id}/apply-encoding",
                            json={"strategies": encoding_choices, "ordinal_orders": ordinal_orders or None},
                        )
                    if r.status_code == 200:
                        result = r.json()
                        st.session_state.analysis = result["analysis"]
                        st.session_state.encoding_recos = None
                        st.success("Encodage appliqué avec succès !")
                        st.dataframe(pd.DataFrame(result["log"]), use_container_width=True)
                    else:
                        st.error(f"Erreur : {r.text}")

    st.divider()
    st.header("6. Visualisation & export")

    summary_response = requests.get(f"{API_URL}/datasets/{dataset_id}/summary")
    if summary_response.status_code == 200:
        summary = summary_response.json()

        st.subheader("Résumé avant / après")
        s1, s2, s3 = st.columns(3)
        s1.metric("Lignes", summary["current"]["n_rows"],
                   delta=summary["current"]["n_rows"] - summary["initial"]["n_rows"])
        s2.metric("Colonnes", summary["current"]["n_columns"],
                   delta=summary["current"]["n_columns"] - summary["initial"]["n_columns"])
        s3.metric(
            "% manquants (global)",
            f"{summary['current']['missing_rate_global']*100:.2f}%",
            delta=f"{(summary['current']['missing_rate_global'] - summary['initial']['missing_rate_global'])*100:.2f} pts",
            delta_color="inverse",
        )

        if summary["treatment_log"]:
            st.subheader("Journal complet des traitements appliqués")
            for step_entry in summary["treatment_log"]:
                with st.expander(f"📋 {step_entry['step'].replace('_', ' ').capitalize()}"):
                    if step_entry["entries"]:
                        st.dataframe(pd.DataFrame(step_entry["entries"]), use_container_width=True)
                    else:
                        st.caption("Aucune modification enregistrée pour cette étape.")
        else:
            st.info("Aucun traitement n'a encore été appliqué.")

        if summary["descriptive_stats"]:
            st.subheader("Statistiques descriptives actuelles (colonnes numériques)")
            st.dataframe(pd.DataFrame(summary["descriptive_stats"]), use_container_width=True)

    st.subheader("Téléchargements")
    dl1, dl2 = st.columns(2)

    csv_response = requests.get(f"{API_URL}/datasets/{dataset_id}/export-csv")
    if csv_response.status_code == 200:
        dl1.download_button(
            label="⬇️ CSV nettoyé",
            data=csv_response.content,
            file_name="dataset_nettoye.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

    code_response = requests.get(f"{API_URL}/datasets/{dataset_id}/generated-code")
    if code_response.status_code == 200:
        dl2.download_button(
            label="⬇️ Script Python reproductible",
            data=code_response.content,
            file_name="pipeline_pretraitement.py",
            mime="text/x-python",
            use_container_width=True,
        )
        with st.expander("👁️ Aperçu du script Python généré"):
            st.code(code_response.content.decode("utf-8"), language="python")

    st.success("✅ Données prêtes pour la modélisation !")
