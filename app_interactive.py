import base64
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PLOTLY_FONT_FAMILY = "GothamBook, Segoe UI, sans-serif"


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


def format_year(value: float) -> str:
    if pd.notna(value):
        return str(int(value))
    return "n/a"


def build_hover(country_iso: str, country_name: str, year_value: float) -> str:
    return f"{country_name} ({country_iso})<br>Год: {format_year(year_value)}"


def style_plotly_figure(fig: go.Figure) -> None:
    fig.update_layout(
        template="simple_white",
        height=560,
        font={"family": PLOTLY_FONT_FAMILY, "size": 12, "color": "#000000"},
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        hovermode="closest",
        legend={"title": None, "font": {"color": "#000000"}},
        title={"font": {"color": "#000000"}},
    )
    fig.update_xaxes(
        showline=True,
        mirror=False,
        linecolor="#000000",
        linewidth=1,
        zeroline=False,
        showgrid=True,
        gridcolor="#D9D9D9",
        gridwidth=0.5,
        griddash="dash",
        title_font={"color": "#000000"},
        tickfont={"color": "#000000"},
    )
    fig.update_yaxes(
        showline=True,
        mirror=False,
        linecolor="#000000",
        linewidth=1,
        zeroline=False,
        showgrid=True,
        gridcolor="#D9D9D9",
        gridwidth=0.5,
        griddash="dash",
        title_font={"color": "#000000"},
        tickfont={"color": "#000000"},
    )


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
    df["country"] = df["country"].astype(str).str.upper().str.strip()
    df["year_num"] = pd.to_numeric(df["year"], errors="coerce")
    df["x"] = pd.to_numeric(df["exp_corrected"], errors="coerce") / 1000.0
    df["y"] = pd.to_numeric(df["sdg3_weighted_mean"], errors="coerce")
    df["country_label"] = df["country"].map(lambda iso: iso3_to_ru.get(iso, iso))

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    fit_mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    has_fit = fit_mask.sum() >= 4

    fig1 = go.Figure()

    base = df[np.isfinite(df["x"]) & np.isfinite(df["y"])].copy()
    base_hover = [
        build_hover(iso, name, year)
        for iso, name, year in zip(base["country"], base["country_label"], base["year_num"])
    ]
    fig1.add_trace(
        go.Scatter(
            x=base["x"],
            y=base["y"],
            mode="markers",
            name="All countries",
            marker={"color": "#A6A6A6", "size": 6, "opacity": 0.35},
            text=base_hover,
            hovertemplate="%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
            showlegend=False,
        )
    )

    if has_fit:
        lx = np.log(x[fit_mask])
        coef = np.polyfit(lx, y[fit_mask], 3)
        x_fit = np.linspace(max(0.01, float(np.nanmin(x[fit_mask]))), min(xmax, float(np.nanmax(x[fit_mask]))), 600)
        y_fit = coef[0] * np.log(x_fit) ** 3 + coef[1] * np.log(x_fit) ** 2 + coef[2] * np.log(x_fit) + coef[3]
        fig1.add_trace(
            go.Scatter(
                x=x_fit,
                y=y_fit,
                mode="lines",
                name="Тренд",
                line={"width": 2, "dash": "dash", "color": "#4472C4"},
                hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
            )
        )

    for c in selected:
        sub = df[df["country"] == c].sort_values("year_num").copy()
        sub = sub[np.isfinite(sub["x"]) & np.isfinite(sub["y"])]
        if sub.empty:
            continue

        label = iso3_to_ru.get(c, c)
        hover_text = [build_hover(c, label, yr) for yr in sub["year_num"]]
        fig1.add_trace(
            go.Scatter(
                x=sub["x"],
                y=sub["y"],
                mode="lines+markers",
                name=label,
                line={"width": 2},
                marker={"size": 7},
                text=hover_text,
                hovertemplate="%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
            )
        )

        if show_labels:
            last = sub.iloc[-1]
            fig1.add_annotation(
                x=float(last["x"]),
                y=float(last["y"]),
                text=f" {format_year(last['year_num'])}",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 10, "color": "#000000"},
            )

    fig1.update_xaxes(
        range=[0, xmax],
        title_text="Расходы на здравоохранение, тыс. долларов США, ППС, в ценах 2024 года",
    )
    fig1.update_yaxes(range=[ymin, ymax], title_text="Индекс здравоохранения")
    fig1.update_layout(
                title={"text": "В правом верхнем углу графика вы можете воспользоваться контроллерами (зум, сохранить картинку и т.д.)"}
            )
    style_plotly_figure(fig1)

    st.plotly_chart(fig1, use_container_width=True, config={"displaylogo": False})


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

            xy00 = {iso: (xv, yv) for iso, xv, yv in zip(c00, x00, y00)}
            xy23 = {iso: (xv, yv) for iso, xv, yv in zip(c23, x23, y23)}
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

            fig2 = go.Figure()

            hover_23 = [build_hover(iso, iso3_to_ru.get(iso, iso), 2023) for iso in c23]
            hover_00 = [build_hover(iso, iso3_to_ru.get(iso, iso), 2000) for iso in c00]

            fig2.add_trace(
                go.Scatter(
                    x=x23,
                    y=y23,
                    mode="markers",
                    name="2023",
                    marker={"size": 7, "color": "#A6A6A6", "opacity": 0.45},
                    text=hover_23,
                    hovertemplate="%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=x00,
                    y=y00,
                    mode="markers",
                    name="2000",
                    marker={"size": 7, "color": "#A6A6A6", "opacity": 0.75, "symbol": "circle-open"},
                    text=hover_00,
                    hovertemplate="%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                )
            )

            fig2.add_trace(
                go.Scatter(
                    x=x23[mask_sel_23],
                    y=y23[mask_sel_23],
                    mode="markers",
                    marker={"size": 9, "color": "#376C8A", "opacity": 0.95},
                    text=[build_hover(iso, iso3_to_ru.get(iso, iso), 2023) for iso in c23[mask_sel_23]],
                    hovertemplate="%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                    showlegend=False,
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=x00[mask_sel_00],
                    y=y00[mask_sel_00],
                    mode="markers",
                    marker={"size": 9, "color": "#376C8A", "opacity": 0.95, "symbol": "circle-open"},
                    text=[build_hover(iso, iso3_to_ru.get(iso, iso), 2000) for iso in c00[mask_sel_00]],
                    hovertemplate="%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                    showlegend=False,
                )
            )

            x_finite_23 = x23[np.isfinite(x23)]
            x_label_shift = 0.2
            if x_finite_23.size > 1:
                x_span_23 = float(np.nanmax(x_finite_23) - np.nanmin(x_finite_23))
                if np.isfinite(x_span_23) and x_span_23 > 0:
                    x_label_shift = max(1, 0.008 * x_span_23)

            for iso in sorted(selected_set & set(common_isos)):
                x0, y0 = xy00[iso]
                x1, y1 = xy23[iso]
                label = iso3_to_ru.get(iso, iso)
                fig2.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode="lines",
                        line={"color": "#376C8A", "width": 1.2},
                        hovertemplate=f"{label} ({iso})<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
                        showlegend=False,
                    )
                )
                fig2.add_annotation(
                    x=x1 + x_label_shift,
                    y=y1,
                    text=f" {iso}",
                    showarrow=False,
                    font={"size": 9, "color": "#000000"},
                    xanchor="left",
                    yanchor="middle",
                )

            fit_mask = np.isfinite(x23) & np.isfinite(y23) & (x23 > 0)
            if fit_mask.sum() >= 4:
                coef = np.polyfit(np.log(x23[fit_mask]), y23[fit_mask], 3)
                x_fit = np.linspace(1.7, float(np.nanmax(x23[fit_mask])), 300)
                y_fit = (
                    coef[0] * np.log(x_fit) ** 3
                    + coef[1] * np.log(x_fit) ** 2
                    + coef[2] * np.log(x_fit)
                    + coef[3]
                )
                fig2.add_trace(
                    go.Scatter(
                        x=x_fit,
                        y=y_fit,
                        mode="lines",
                        name="Тренд (2023)",
                        line={"width": 2, "dash": "dash", "color": "#4472C4"},
                        hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                    )
                )

            fig2.update_xaxes(
                title_text="ВНД, тыс. долларов США, ППС, в ценах 2024 года",
                range=[x_lim_v2[0], x_lim_v2[1]],
            )
            fig2.update_yaxes(title_text="ОПЖ", range=[y_lim_v2[0], y_lim_v2[1]])
            fig2.update_layout(
                title={"text": "Тренд основан на выборке 2023 года, выбранные страны соединены"}
            )
            style_plotly_figure(fig2)

            st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False})
