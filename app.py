import io
import os
import base64
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


@st.cache_data
def load_expenditure(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_le_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed") or str(c).strip() == ""]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df


@st.cache_data
def load_country_name_map(path: str) -> dict:
    """ISO3 -> country display name map from keys.csv (id, name)."""
    df = pd.read_csv(path)
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed") or str(c).strip() == ""]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    if "id" not in df.columns or "name" not in df.columns:
        return {}

    df["id"] = df["id"].astype(str).str.upper().str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df = df[df["id"].str.len() == 3]
    return dict(zip(df["id"], df["name"]))


def build_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


@st.cache_resource
def configure_chart_font() -> fm.FontProperties:
    """Register Gotham Book and return a reusable FontProperties."""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    gotham_path = os.path.join(app_dir, "Gotham-Book.ttf")

    if os.path.exists(gotham_path):
        fm.fontManager.addfont(gotham_path)
        gotham_name = fm.FontProperties(fname=gotham_path).get_name()
        plt.rcParams["font.family"] = gotham_name
        plt.rcParams["font.sans-serif"] = [gotham_name]
        return fm.FontProperties(fname=gotham_path)

    return fm.FontProperties()


@st.cache_data
def build_streamlit_font_css() -> str:
    """Embed Gotham Book for app text while preserving icon fonts."""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    gotham_path = os.path.join(app_dir, "Gotham-Book.ttf")
    if not os.path.exists(gotham_path):
        return ""

    with open(gotham_path, "rb") as font_file:
        font_b64 = base64.b64encode(font_file.read()).decode("utf-8")

    return f"""
    <style>
      @font-face {{
        font-family: "GothamBook";
        src: url(data:font/ttf;base64,{font_b64}) format("truetype");
        font-weight: 400;
        font-style: normal;
      }}

      :root {{
        --app-font: "GothamBook", "Segoe UI", sans-serif;
      }}

      /* Apply Gotham globally */
      .stApp,
      .stApp * {{
        font-family: var(--app-font) !important;
      }}

      /* Restore icon fonts (Streamlit/BaseWeb/Material icons) */
      [data-testid="stIconMaterial"],
      [data-testid="stIconMaterial"] *,
      .material-symbols-rounded,
      .material-symbols-outlined,
      .material-icons,
      .material-icons-round,
      .material-icons-outlined,
      [class*="material-symbols"],
      [class*="material-icons"] {{
        font-family: "Material Symbols Rounded", "Material Symbols Outlined",
                     "Material Icons", sans-serif !important;
        font-style: normal !important;
        font-weight: 400 !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        direction: ltr !important;
        -webkit-font-feature-settings: "liga" !important;
        -webkit-font-smoothing: antialiased !important;
      }}
    </style>
    """


def apply_streamlit_font() -> None:
    css = build_streamlit_font_css()
    if css:
        st.markdown(css, unsafe_allow_html=True)


FONT_PROP = configure_chart_font()


def apply_axis_font(ax, font_prop: fm.FontProperties) -> None:
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)


st.set_page_config(page_title="Health visuals", layout="wide")
apply_streamlit_font()
st.title("Аналитика для здравоохранения")


# -----------------------------------------------------------------------------
# Visual 1: SDG3 vs health expenditure trajectory
# -----------------------------------------------------------------------------
st.header("Индекс здравоохранения против расходов на здравоохранение, динамика 2000-2023")

try:
    iso3_to_ru = load_country_name_map("keys.csv")
except FileNotFoundError:
    iso3_to_ru = {}

try:
    expenditure_ppp_viz = load_expenditure("expenditure_ppp_viz.csv")
except FileNotFoundError:
    st.error("File not found: expenditure_ppp_viz.csv")
    expenditure_ppp_viz = pd.DataFrame()

if not expenditure_ppp_viz.empty:
    st.sidebar.header("Индекс здравоохранения против расходов на здравоохранение, динамика 2000-2023, фильтры")

    all_countries = sorted(
        expenditure_ppp_viz["country"].dropna().astype(str).unique().tolist(),
        key=lambda iso: iso3_to_ru.get(iso, iso),
    )
    if iso3_to_ru:
        all_countries = [iso for iso in all_countries if iso in iso3_to_ru]
    default_sel = [c for c in ["KAZ", "UZB"] if c in all_countries]

    selected = st.sidebar.multiselect(
        "Страны",
        options=all_countries,
        format_func=lambda iso: iso3_to_ru.get(iso, iso),
        default=default_sel if default_sel else (all_countries[:1] if all_countries else []),
        key="v1_selected_countries",
    )

    show_labels = st.sidebar.checkbox("Показывать последний год", value=True, key="v1_show_labels")
    xmax = st.sidebar.slider("Лимиты по оси X", 0.5, 15.0, 3.5, 0.1, key="v1_xmax")
    ymin, ymax = st.sidebar.slider("Лимиты по оси Y", 0, 100, (20, 90), 1, key="v1_ylim")

    df = expenditure_ppp_viz.copy()

    x = pd.to_numeric(df["exp_corrected"], errors="coerce").to_numpy() / 1000.0
    y = pd.to_numeric(df["sdg3_weighted_mean"], errors="coerce").to_numpy()

    fit_mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    has_fit = fit_mask.sum() >= 4

    fig1, ax1 = plt.subplots(figsize=(12, 6))

    ax1.scatter(x, y, alpha=0.35, s=15, color="#A6A6A6")

    if has_fit:
        lx = np.log(x[fit_mask])
        coef = np.polyfit(lx, y[fit_mask], 3)
        x_fit = np.linspace(max(0.01, np.nanmin(x[fit_mask])), min(xmax, np.nanmax(x[fit_mask])), 600)
        y_fit = coef[0] * np.log(x_fit) ** 3 + coef[1] * np.log(x_fit) ** 2 + coef[2] * np.log(x_fit) + coef[3]
        ax1.plot(x_fit, y_fit, linewidth=1.5, linestyle="--", color="#4472C4")

    for c in selected:
        sub = df[df["country"] == c].sort_values("year")
        if sub.empty:
            continue

        x_c = pd.to_numeric(sub["exp_corrected"], errors="coerce").to_numpy() / 1000.0
        y_c = pd.to_numeric(sub["sdg3_weighted_mean"], errors="coerce").to_numpy()

        label = iso3_to_ru.get(c, c)
        ax1.plot(x_c, y_c, linewidth=1.8, marker="o", markersize=4, label=label, zorder=3)

        if show_labels and len(sub) > 0:
            ax1.text(
                x_c[-1],
                y_c[-1],
                f" {int(sub['year'].iloc[-1])}",
                fontsize=10,
                va="center",
                fontproperties=FONT_PROP,
            )

    ax1.set_xlim(0, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.set_ylabel("Индекс здравоохранения", fontsize=11, fontproperties=FONT_PROP)
    ax1.set_xlabel(
        "Расходы на здравоохранение, тыс. долларов США, ППС, в ценах 2024 года",
        fontsize=11,
        fontproperties=FONT_PROP,
    )
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    apply_axis_font(ax1, FONT_PROP)

    if selected:
        ax1.legend(frameon=False, prop=FONT_PROP)

    fig1_png = build_png_bytes(fig1)
    st.pyplot(fig1, clear_figure=True)

    st.download_button(
        "Скачать в png",
        data=fig1_png,
        file_name="SDGHealthvsExpDyn.png",
        mime="image/png",
        key="v1_download_png",
    )


# -----------------------------------------------------------------------------
# Visual 2: LE vs GNIPPPconst (2000 and 2023)
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("ОПЖ против ВНД")

required_x_col = "GNIPPPconst"

try:
    le_raw = load_le_dataset("LEvsGNIPPPconst2000_2023.csv")
except FileNotFoundError:
    st.error("File not found: LEvsGNIPPPconst2000_2023.csv")
    le_raw = pd.DataFrame()

if not le_raw.empty:
    year_col = "year" if "year" in le_raw.columns else ("Year" if "Year" in le_raw.columns else None)

    if (
        year_col is None
        or "Country Code" not in le_raw.columns
        or "LE" not in le_raw.columns
        or required_x_col not in le_raw.columns
    ):
        st.error("LE file must contain: Country Code, year/Year, LE, and GNIPPPconst.")
    else:
        le_df = le_raw.copy()
        le_df["Country Code"] = le_df["Country Code"].astype(str).str.upper().str.strip()
        le_df[year_col] = pd.to_numeric(le_df[year_col], errors="coerce")
        le_df["LE"] = pd.to_numeric(le_df["LE"], errors="coerce")
        le_df[required_x_col] = pd.to_numeric(le_df[required_x_col], errors="coerce")
        le_df = le_df.dropna(subset=["Country Code", year_col, "LE", required_x_col])
        le_df = le_df[le_df["Country Code"].str.len() == 3]

        d00 = le_df[le_df[year_col] == 2000].copy()
        d23 = le_df[le_df[year_col] == 2023].copy()

        if d00.empty or d23.empty:
            st.error("Could not find both year 2000 and 2023 in LE file.")
        else:
            # x-axis in thousands
            d00["x"] = d00[required_x_col] / 1000.0
            d23["x"] = d23[required_x_col] / 1000.0

            d00 = d00.groupby("Country Code", as_index=False).agg(x=("x", "mean"), LE=("LE", "mean"))
            d23 = d23.groupby("Country Code", as_index=False).agg(x=("x", "mean"), LE=("LE", "mean"))

            x00 = d00["x"].to_numpy()
            y00 = d00["LE"].to_numpy()
            c00 = d00["Country Code"].astype(str).to_numpy()

            x23 = d23["x"].to_numpy()
            y23 = d23["LE"].to_numpy()
            c23 = d23["Country Code"].astype(str).to_numpy()

            xy00 = {iso: (x, y) for iso, x, y in zip(c00, x00, y00)}
            xy23 = {iso: (x, y) for iso, x, y in zip(c23, x23, y23)}
            common_isos = sorted(set(xy00) & set(xy23), key=lambda iso: iso3_to_ru.get(iso, iso))
            if iso3_to_ru:
                common_isos = [iso for iso in common_isos if iso in iso3_to_ru]

            st.sidebar.header("ОПЖ против ВНД, фильтры")
            default_connect = [c for c in ["KAZ", "HUN", "SVK", "TUR"] if c in common_isos]
            if not default_connect and common_isos:
                default_connect = common_isos[:5]

            selected_connectors = st.sidebar.multiselect(
                "Страны",
                options=common_isos,
                format_func=lambda iso: iso3_to_ru.get(iso, iso),
                default=default_connect,
                key="v2_connectors",
            )
            selected_set = set(selected_connectors)
            x_lim_v2 = st.sidebar.slider(
                "Лимиты по оси X",
                min_value=0.0,
                max_value=120.0,
                value=(0.0, 120.0),
                step=0.5,
                key="v2_xlim",
            )
            y_lim_v2 = st.sidebar.slider(
                "Лимиты по оси Y",
                min_value=40.0,
                max_value=90.0,
                value=(60.0, 85.0),
                step=0.5,
                key="v2_ylim",
            )
            selected_list = list(selected_set)
            mask_sel_23 = np.isin(c23, selected_list)
            mask_sel_00 = np.isin(c00, selected_list)

            fig2, ax2 = plt.subplots(figsize=(12, 6))

            # Points for both years
            ax2.scatter(x23, y23, s=24, color="#A6A6A6", alpha=0.45, zorder=2, label="2023")
            ax2.scatter(
                x00, y00, s=24, facecolors="none", edgecolors="#A6A6A6",
                linewidth=0.8, alpha=0.7, zorder=1, label="2000"
            )

            # Selected countries overlay in blue
            ax2.scatter(x23[mask_sel_23], y23[mask_sel_23], s=32, color="#376C8A", alpha=0.95, zorder=4)
            ax2.scatter(
                x00[mask_sel_00], y00[mask_sel_00], s=32, facecolors="none",
                edgecolors="#376C8A", linewidth=1.1, alpha=0.95, zorder=4
            )

            # Small right shift for code labels to reduce overlap with markers.
            x_finite_23 = x23[np.isfinite(x23)]
            x_label_shift = 0.2
            if x_finite_23.size > 1:
                x_span_23 = float(np.nanmax(x_finite_23) - np.nanmin(x_finite_23))
                if np.isfinite(x_span_23) and x_span_23 > 0:
                    x_label_shift = max(1, 0.008 * x_span_23)

            # Connect only selected countries
            for iso in sorted(selected_set & set(common_isos)):
                x0, y0 = xy00[iso]
                x1, y1 = xy23[iso]
                ax2.plot([x0, x1], [y0, y1], color="#376C8A", linewidth=1.0, alpha=0.75, zorder=3)
                ax2.text(
                    x1 + x_label_shift,
                    y1,
                    f" {iso}",
                    fontsize=8,
                    color="#376C8A",
                    zorder=6,
                    va="center",
                    fontproperties=FONT_PROP,
                )

            # Trendline on ALL 2023 observations
            fit_mask = np.isfinite(x23) & np.isfinite(y23) & (x23 > 0)
            if fit_mask.sum() >= 4:
                coef = np.polyfit(np.log(x23[fit_mask]), y23[fit_mask], 3)
                x_fit = np.linspace(np.nanmin(x23[fit_mask]), np.nanmax(x23[fit_mask]), 300)
                y_fit = (
                    coef[0] * np.log(x_fit) ** 3
                    + coef[1] * np.log(x_fit) ** 2
                    + coef[2] * np.log(x_fit)
                    + coef[3]
                )
                ax2.plot(x_fit, y_fit, linewidth=1.4, linestyle="--", color="#4472C4", zorder=0, label="Тренд (2023)")

            ax2.set_xlabel("ВНД, тыс. долларов США, ППС, в ценах 2024 года", fontsize=11, fontproperties=FONT_PROP)
            ax2.set_ylabel("ОПЖ", fontsize=11, fontproperties=FONT_PROP)
            ax2.set_title(
                "Тренд основан на выборке 2023 года, выбранные страны соединены",
                fontsize=12,
                fontproperties=FONT_PROP,
            )

            ax2.set_xlim(x_lim_v2[0], x_lim_v2[1])
            ax2.set_ylim(y_lim_v2[0], y_lim_v2[1])

            ax2.grid(True, which="both", linestyle="--", linewidth=0.5, color="#D9D9D9")
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            apply_axis_font(ax2, FONT_PROP)

            handles, labels = ax2.get_legend_handles_labels()
            dedup = {}
            for h, l in zip(handles, labels):
                if l and l not in dedup:
                    dedup[l] = h
            if dedup:
                ax2.legend(dedup.values(), dedup.keys(), frameon=False, loc="best", prop=FONT_PROP)

            fig2_png = build_png_bytes(fig2)
            st.pyplot(fig2, clear_figure=True)
         

            st.download_button(
                "Скачать в png",
                data=fig2_png,
                file_name="LEvsGNI_2000_2023.png",
                mime="image/png",
                key="v2_download_png",
            )
