
import streamlit as st
import pandas as pd
import datetime as dt
import requests

st.set_page_config(page_title="MG Auto Dati (football-data.org)", layout="wide")
st.title("MG Auto Dati – stagione corrente (football-data.org)")

st.markdown(
    """
Questa app prende i dati da sola da football-data.org (stagione corrente) e calcola:

- Produzione gol (G0..G4+) della squadra scelta su tutta la stagione (n match usati indicato)
- Gol subiti casa/trasferta (G0..G4+) dell'avversaria nelle ultime 10 partite coerenti
  (solo casa se gioca in casa, solo trasferta se gioca fuori).
  Se sono meno di 10 partite, usa quelle disponibili e indica quante sono.

Serve un token API in Secrets: FOOTBALL_DATA_TOKEN="...".
"""
)

TOKEN = st.secrets.get("FOOTBALL_DATA_TOKEN", "").strip()
if not TOKEN:
    st.error("Manca FOOTBALL_DATA_TOKEN nei Secrets di Streamlit.")
    st.stop()

BASE = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": TOKEN}

COMPETITIONS = {
    "Serie A (SA)": "SA",
    "Serie B (SB)": "SB",
    "Premier League (PL)": "PL",
    "LaLiga (PD)": "PD",
    "Bundesliga (BL1)": "BL1",
    "Ligue 1 (FL1)": "FL1",
    "Eredivisie (DED)": "DED",
    "Primeira Liga – Portogallo (PPL)": "PPL",
}


@st.cache_data(show_spinner=False, ttl=60*30)
def api_get(path: str, params: dict | None = None) -> dict:
    """GET verso football-data.org con cache e retry/backoff su 429."""
    import time as _time
    url = f"{BASE}{path}"
    max_tries = 6
    last_err = None

    for i in range(max_tries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=30)
        except Exception as e:
            last_err = e
            _time.sleep(min(2*(i+1), 10))
            continue

        if r.status_code == 200:
            return r.json()

        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            try:
                wait = int(retry_after) if retry_after else 12*(i+1)
            except Exception:
                wait = 12*(i+1)
            wait = min(max(wait, 8), 60)
            if i == 0:
                st.warning("Rate limit API (429): rallento automaticamente e riprovo…")
            _time.sleep(wait)
            last_err = RuntimeError("Rate limit (429)")
            continue

        if r.status_code >= 400:
            raise RuntimeError(f"Errore API {r.status_code}: {r.text[:300]}")

    st.error("Rate limit/errore rete persistente. Riprova tra 1–2 minuti.")
    raise RuntimeError(str(last_err) if last_err else "Errore API persistente")


@st.cache_data(show_spinner=False, ttl=60*15)
def get_competition_matches(comp_code: str, date_from: str, date_to: str) -> pd.DataFrame:
    params = {"dateFrom": date_from, "dateTo": date_to}
    data = api_get(f"/competitions/{comp_code}/matches", params=params)
    matches = data.get("matches", [])
    rows = []
    for m in matches:
        home = m.get("homeTeam", {})
        away = m.get("awayTeam", {})
        score = m.get("score", {})
        full = score.get("fullTime", {}) if isinstance(score, dict) else {}
        rows.append({
            "match_id": m.get("id"),
            "utcDate": m.get("utcDate"),
            "status": m.get("status"),
            "home_id": home.get("id"),
            "home_name": home.get("name"),
            "away_id": away.get("id"),
            "away_name": away.get("name"),
            "home_ft": full.get("home"),
            "away_ft": full.get("away"),
            "matchday": m.get("matchday"),
        })
    df = pd.DataFrame(rows)
    if not df.empty and "utcDate" in df.columns:
        df["utcDate"] = pd.to_datetime(df["utcDate"], errors="coerce", utc=True)
    return df

@st.cache_data(show_spinner=False, ttl=60*60)
def get_team_season_matches(team_id: int, comp_code: str) -> pd.DataFrame:
    data = api_get(f"/teams/{team_id}/matches", params={"competitions": comp_code})
    matches = data.get("matches", [])
    rows = []
    for m in matches:
        home = m.get("homeTeam", {})
        away = m.get("awayTeam", {})
        score = m.get("score", {})
        full = score.get("fullTime", {}) if isinstance(score, dict) else {}
        rows.append({
            "match_id": m.get("id"),
            "utcDate": m.get("utcDate"),
            "status": m.get("status"),
            "home_id": home.get("id"),
            "away_id": away.get("id"),
            "home_name": home.get("name"),
            "away_name": away.get("name"),
            "home_ft": full.get("home"),
            "away_ft": full.get("away"),
        })
    df = pd.DataFrame(rows)
    if not df.empty and "utcDate" in df.columns:
        df["utcDate"] = pd.to_datetime(df["utcDate"], errors="coerce", utc=True)
    return df

def goals_for_in_match(row: pd.Series, team_id: int):
    if row.get("status") != "FINISHED":
        return None
    if row.get("home_id") == team_id:
        return row.get("home_ft")
    if row.get("away_id") == team_id:
        return row.get("away_ft")
    return None

def goals_conceded_in_match(row: pd.Series, team_id: int):
    if row.get("status") != "FINISHED":
        return None
    if row.get("home_id") == team_id:
        return row.get("away_ft")
    if row.get("away_id") == team_id:
        return row.get("home_ft")
    return None

def bucket_0_4p(x: int) -> str:
    return f"G{x}" if x <= 3 else "G4+"

def dist_table(counts: pd.Series, total: int) -> pd.DataFrame:
    order = ["G0","G1","G2","G3","G4+"]
    out = []
    for k in order:
        c = int(counts.get(k, 0))
        p = (c / total) if total > 0 else 0.0
        out.append({"Bucket": k, "Count": c, "Percent": p})
    return pd.DataFrame(out)

def dist_compare_context(total_df: pd.DataFrame, ctx_df: pd.DataFrame, ctx_label: str) -> pd.DataFrame:
    """Tabella comparativa: % gol fatti Totale stagione vs contesto (Casa/Trasferta)."""
    order = ["G0","G1","G2","G3","G4+"]
    tot_counts = total_df["bucket_gf"].value_counts() if total_df is not None and len(total_df) else pd.Series(dtype=int)
    tot_n = int(len(total_df)) if total_df is not None else 0

    ctx_counts = ctx_df["bucket_gf"].value_counts() if ctx_df is not None and len(ctx_df) else pd.Series(dtype=int)
    ctx_n = int(len(ctx_df)) if ctx_df is not None else 0

    rows = []
    for k in order:
        rows.append({
            "Gol": k,
            "Totale": (float(tot_counts.get(k, 0)) / tot_n) if tot_n else 0.0,
            ctx_label: (float(ctx_counts.get(k, 0)) / ctx_n) if ctx_n else None,
        })
    return pd.DataFrame(rows)


col1, col2 = st.columns([1, 2])

with col1:
    comp_label = st.selectbox("Campionato", list(COMPETITIONS.keys()))
    comp = COMPETITIONS[comp_label]
    days_ahead = st.slider("Finestra partite (giorni avanti)", 1, 14, 7)
    today = dt.date.today()
    date_from = today.isoformat()
    date_to = (today + dt.timedelta(days=int(days_ahead))).isoformat()

with st.spinner("Carico partite..."):
    matches_df = get_competition_matches(comp, date_from, date_to)

if matches_df.empty:
    st.warning("Nessuna partita trovata nella finestra scelta. Prova ad aumentare i giorni.")
    st.stop()

upcoming = matches_df[matches_df["status"].isin(["SCHEDULED", "TIMED", "IN_PLAY", "PAUSED"])].copy()
if upcoming.empty:
    upcoming = matches_df.copy()

def match_label(r):
    dt_str = ""
    if pd.notna(r.get("utcDate")):
        try:
            dt_str = r["utcDate"].tz_convert("Europe/Rome").strftime("%d/%m %H:%M")
        except Exception:
            dt_str = str(r.get("utcDate"))
    md = r.get("matchday")
    md_str = f"MD {md} - " if pd.notna(md) else ""
    return f"{md_str}{dt_str} | {r['home_name']} vs {r['away_name']}"

upcoming = upcoming.sort_values("utcDate") if "utcDate" in upcoming.columns else upcoming
labels = [match_label(r) for _, r in upcoming.iterrows()]

with col2:
    st.subheader("Seleziona partita")
    idx = st.selectbox("Partita", list(range(len(labels))), format_func=lambda i: labels[i])

sel = upcoming.iloc[int(idx)]
home_id = int(sel["home_id"])
away_id = int(sel["away_id"])
home_name = sel["home_name"]
away_name = sel["away_name"]

st.divider()
st.subheader("Calcolo distribuzioni")

N_LAST = 10

with st.spinner("Scarico match stagione e calcolo..."):
    home_season = get_team_season_matches(home_id, comp)
    away_season = get_team_season_matches(away_id, comp)

    hs = home_season[home_season["status"] == "FINISHED"].copy()
    aw = away_season[away_season["status"] == "FINISHED"].copy()

    hs["gf"] = hs.apply(lambda r: goals_for_in_match(r, home_id), axis=1)
    aw["gf"] = aw.apply(lambda r: goals_for_in_match(r, away_id), axis=1)
    hs = hs.dropna(subset=["gf"])
    aw = aw.dropna(subset=["gf"])

    hs["bucket_gf"] = hs["gf"].astype(int).apply(bucket_0_4p)
    aw["bucket_gf"] = aw["gf"].astype(int).apply(bucket_0_4p)

    
    # --- SPLIT CASA/TRASFERTA: gol FATTI (stagione) ---
    hs_home_gf = hs[hs["home_id"] == home_id].copy()
    hs_away_gf = hs[hs["away_id"] == home_id].copy()
    aw_home_gf = aw[aw["home_id"] == away_id].copy()
    aw_away_gf = aw[aw["away_id"] == away_id].copy()

    for _df in [hs_home_gf, hs_away_gf, aw_home_gf, aw_away_gf]:
        if not _df.empty:
            _df["bucket_gf"] = _df["gf"].astype(int).apply(bucket_0_4p)

# --- Indicatori trend (ultime 6 vs stagione) sui gol FATTI ---
    def _trend_metrics(team_df: pd.DataFrame, team_label: str) -> dict:
        out = {
            "Squadra": team_label,
            "Match stagione (FINISHED)": int(len(team_df)),
            "Media gol stagione": float(team_df["gf"].mean()) if len(team_df) else 0.0,
            "Match usati ultime 6": int(min(6, len(team_df))),
            "Media gol ultime 6": 0.0,
            "Delta (ult6 - stag)": 0.0,
            "Evento estremo (ult6)": "",
            "Estremi (ult6)": 0,
            "Stato": "DATI INSUFFICIENTI",
        }
        if len(team_df) < 3:
            return out

        recent = team_df.sort_values("utcDate", ascending=False).head(6).copy()
        m6 = float(recent["gf"].mean()) if len(recent) else 0.0
        delta = m6 - float(out["Media gol stagione"])
        out["Media gol ultime 6"] = m6
        out["Delta (ult6 - stag)"] = delta

        if delta >= 0:
            out["Evento estremo (ult6)"] = "3+"
            out["Estremi (ult6)"] = int((recent["gf"] >= 3).sum())
        else:
            out["Evento estremo (ult6)"] = "0"
            out["Estremi (ult6)"] = int((recent["gf"] == 0).sum())

        if len(recent) < 6:
            out["Stato"] = "DATI INSUFFICIENTI"
            return out

        abs_delta = abs(delta)
        extremes = out["Estremi (ult6)"]
        if abs_delta >= 0.7 and extremes >= 3:
            out["Stato"] = "CAMBIO CONFERMATO"
        elif abs_delta >= 0.4:
            out["Stato"] = "WARNING"
        else:
            out["Stato"] = "NORMAL"
        return out

    home_tr = _trend_metrics(hs, home_name)
    away_tr = _trend_metrics(aw, away_name)

    # Gol subiti coerenti (ultime 10)
    home_home = hs[hs["home_id"] == home_id].copy()
    home_home["ga"] = home_home.apply(lambda r: goals_conceded_in_match(r, home_id), axis=1)
    home_home = home_home.dropna(subset=["ga"]).sort_values("utcDate", ascending=False).head(N_LAST)
    home_home["bucket_ga"] = home_home["ga"].astype(int).apply(bucket_0_4p)

    away_away = aw[aw["away_id"] == away_id].copy()
    away_away["ga"] = away_away.apply(lambda r: goals_conceded_in_match(r, away_id), axis=1)
    away_away = away_away.dropna(subset=["ga"]).sort_values("utcDate", ascending=False).head(N_LAST)
    away_away["bucket_ga"] = away_away["ga"].astype(int).apply(bucket_0_4p)


st.subheader("Indicatori trend (automatici) – gol fatti: ultime 6 vs stagione")

def _badge(s: str) -> str:
    if s == "CAMBIO CONFERMATO":
        return "🔴 CAMBIO CONFERMATO"
    if s == "WARNING":
        return "🟡 WARNING"
    if s == "NORMAL":
        return "🟢 NORMAL"
    return "⚪ DATI INSUFFICIENTI"

trend_df = pd.DataFrame([home_tr, away_tr])
trend_df["Stato"] = trend_df["Stato"].apply(_badge)

st.dataframe(
    trend_df[[
        "Squadra",
        "Match stagione (FINISHED)",
        "Media gol stagione",
        "Match usati ultime 6",
        "Media gol ultime 6",
        "Delta (ult6 - stag)",
        "Evento estremo (ult6)",
        "Estremi (ult6)",
        "Stato",
    ]].style.format({
        "Media gol stagione": "{:.2f}",
        "Media gol ultime 6": "{:.2f}",
        "Delta (ult6 - stag)": "{:+.2f}",
    }),
    use_container_width=True,
    hide_index=True
)
st.caption("Regole: 🔴 CAMBIO CONFERMATO se |Δ| ≥ 0.7 e l’evento estremo (3+ se Δ≥0, altrimenti 0) esce ≥ 3 volte nelle ultime 6. 🟡 WARNING se |Δ| ≥ 0.4.")
st.divider()

c1, c2 = st.columns(2)
with c1:
    st.markdown(f"### {home_name} – Gol fatti (Totale vs Casa)")
    df_h_cmp = dist_compare_context(hs, hs_home_gf, "Casa")
    st.dataframe(
        df_h_cmp.style.format({"Totale":"{:.1%}", "Casa":"{:.1%}"}),
        use_container_width=True,
        hide_index=True
    )
    st.caption(f"Match usati: Totale={len(hs)} | Casa={len(hs_home_gf)}")

with c2:
    st.markdown(f"### {away_name} – Gol subiti in trasferta (ultime {min(N_LAST, len(away_away))} partite)")
    st.dataframe(dist_table(away_away["bucket_ga"].value_counts(), len(away_away)).style.format({"Percent":"{:.1%}"}),
                 use_container_width=True, hide_index=True)

st.divider()

c3, c4 = st.columns(2)
with c3:
    st.markdown(f"### {away_name} – Gol fatti (Totale vs Trasferta)")
    df_a_cmp = dist_compare_context(aw, aw_away_gf, "Trasferta")
    st.dataframe(
        df_a_cmp.style.format({"Totale":"{:.1%}", "Trasferta":"{:.1%}"}),
        use_container_width=True,
        hide_index=True
    )
    st.caption(f"Match usati: Totale={len(aw)} | Trasferta={len(aw_away_gf)}")

with c4:
    st.markdown(f"### {home_name} – Gol subiti in casa (ultime {min(N_LAST, len(home_home))} partite)")
    st.dataframe(dist_table(home_home["bucket_ga"].value_counts(), len(home_home)).style.format({"Percent":"{:.1%}"}),
                 use_container_width=True, hide_index=True)



st.divider()

# ===========================
# CHECKLIST WIREFRAME (NO H2H)
# ===========================

st.subheader("Checklist guidata (wireframe)")

# Scelta squadra da valutare (multigol squadra specifica)
team_choice = st.radio("Squadra da valutare (mercato squadra specifica)", ["Casa", "Trasferta"], horizontal=True)

# Costruisci distribuzioni % per gol fatti della squadra scelta
def _pct_dict_from_buckets(series_buckets: pd.Series) -> dict:
    order = ["G0","G1","G2","G3","G4+"]
    total = int(series_buckets.sum()) if series_buckets is not None else 0
    out = {k: 0.0 for k in order}
    if total <= 0:
        return out
    for k in order:
        out[k] = float(series_buckets.get(k, 0)) / total
    return out

if team_choice == "Casa":
    team_name = home_name
    opp_name = away_name
    team_dist = _pct_dict_from_buckets(hs["bucket_gf"].value_counts())
    opp_conc = _pct_dict_from_buckets(away_away["bucket_ga"].value_counts())
    trend_row = home_tr
else:
    team_name = away_name
    opp_name = home_name
    team_dist = _pct_dict_from_buckets(aw["bucket_gf"].value_counts())
    opp_conc = _pct_dict_from_buckets(home_home["bucket_ga"].value_counts())
    trend_row = away_tr

# Distribuzioni CONTEXT (gol fatti casa/trasferta coerenti col match) per mercati di PARTITA (Under/BTTS)
# Fallback su stagione totale se split insufficiente.
CTX_MIN_MATCHES = 6

def _ctx_or_total(split_df: pd.DataFrame, total_df: pd.DataFrame) -> pd.DataFrame:
    return split_df if split_df is not None and len(split_df) >= CTX_MIN_MATCHES else total_df

if team_choice == "Casa":
    team_for_ctx_df = _ctx_or_total(hs_home_gf, hs)          # gol fatti Inter in casa
    opp_for_ctx_df  = _ctx_or_total(aw_away_gf, aw)          # gol fatti avversaria in trasferta
    team_conc_ctx   = _pct_dict_from_buckets(home_home["bucket_ga"].value_counts())   # subiti casa (ult N)
    opp_conc_ctx    = _pct_dict_from_buckets(away_away["bucket_ga"].value_counts())   # subiti trasferta (ult N)
else:
    team_for_ctx_df = _ctx_or_total(aw_away_gf, aw)          # gol fatti squadra in trasferta
    opp_for_ctx_df  = _ctx_or_total(hs_home_gf, hs)          # gol fatti avversaria in casa
    team_conc_ctx   = _pct_dict_from_buckets(away_away["bucket_ga"].value_counts())   # subiti trasferta (ult N)
    opp_conc_ctx    = _pct_dict_from_buckets(home_home["bucket_ga"].value_counts())   # subiti casa (ult N)

team_for_ctx = _pct_dict_from_buckets(team_for_ctx_df["bucket_gf"].value_counts())
opp_for_ctx  = _pct_dict_from_buckets(opp_for_ctx_df["bucket_gf"].value_counts())


# --- STEP 0 inputs ---
c0a, c0b = st.columns([2,1])
with c0a:
    quota = st.number_input("Quota (facoltativa, per filtro Step 0)", min_value=1.01, max_value=100.0, value=1.62, step=0.01)
with c0b:
    big_match = st.checkbox("Big match caotico", value=False)
motivazioni = st.checkbox("Motivazioni anomale (derby/coppa/ultima giornata)", value=False)

def _is_flat(d: dict) -> bool:
    # distribuzione "piatta" se nessuna coppia adiacente supera 55% e max < 35%
    vals = [d.get("G0",0),d.get("G1",0),d.get("G2",0),d.get("G3",0),d.get("G4+",0)]
    mx = max(vals) if vals else 0
    pairs = [d.get("G0",0)+d.get("G1",0), d.get("G1",0)+d.get("G2",0), d.get("G2",0)+d.get("G3",0)]
    return (mx < 0.35) and all(p < 0.55 for p in pairs)

# Rendering stabile (NO f-strings multiline)
def render_result(r: dict):
    text = (
        r.get("badge","")
        + " **"
        + r.get("name","")
        + " — "
        + r.get("label","")
        + "**\n"
        + "Perde se: "
        + r.get("lose_if","")
    )
    st.markdown(text)

def step_box(title: str, rows: list, status: str, kind: str="info"):
    # kind: info/success/warning/error
    body = "\n".join(rows)
    box = f"### {title}\n{body}\n\n**STATUS: {status}**"
    if kind == "success":
        st.success(box)
    elif kind == "warning":
        st.warning(box)
    elif kind == "error":
        st.error(box)
    else:
        st.info(box)

# ---------- STEP 0 ----------
pref_rows = []
pref_rows.append(f"Quota ≥ 1.62: {'✔️' if quota >= 1.62 else '❌'} (quota={quota:.2f})")
pref_rows.append(f"Big match caotico: {'❌' if big_match else '✔️ NO'}")
pref_rows.append(f"Motivazioni anomale: {'❌' if motivazioni else '✔️ NO'}")
pref_rows.append(f"Distribuzione non piatta: {'✔️' if not _is_flat(team_dist) else '❌'}")

step0_ok = (quota >= 1.62) and (not big_match) and (not motivazioni) and (not _is_flat(team_dist))
step_box("STEP 0 — PRE-FILTRO", pref_rows, "OK" if step0_ok else "NO BET", "success" if step0_ok else "error")

if not step0_ok:
    st.stop()

# ---------- STEP 1 ----------
g01 = team_dist["G0"] + team_dist["G1"]
g12 = team_dist["G1"] + team_dist["G2"]
g23 = team_dist["G2"] + team_dist["G3"]

bar = None
if g01 >= 0.55:
    bar = "BASSO"
elif g12 >= 0.55:
    bar = "MEDIO"
elif g23 >= 0.55:
    bar = "ALTO"

step1_rows = [
    f"G0+G1 = {g01:.0%}",
    f"G1+G2 = {g12:.0%}",
    f"G2+G3 = {g23:.0%}",
    f"➜ BARICENTRO: {bar if bar else 'NESSUNO'}",
]
step1_ok = bar is not None
step_box("STEP 1 — BARICENTRO (gol fatti)", step1_rows, "OK" if step1_ok else "NO BET", "success" if step1_ok else "error")
if not step1_ok:
    st.stop()

# ---------- STEP 2 ----------

def compute_bar(dist: dict):
    g01 = dist["G0"] + dist["G1"]
    g12 = dist["G1"] + dist["G2"]
    g23 = dist["G2"] + dist["G3"]
    if g01 >= 0.55:
        return "BASSO", g01, g12, g23
    if g12 >= 0.55:
        return "MEDIO", g01, g12, g23
    if g23 >= 0.55:
        return "ALTO", g01, g12, g23
    return None, g01, g12, g23

def ranges_for_bar(bar: str):
    if bar == "BASSO":
        return ["0–1", "0–2"]
    if bar == "MEDIO":
        return ["1–2", "1–3"]
    if bar == "ALTO":
        return ["2–3", "2–4"]
    return []

ranges = []
if bar == "BASSO":
    ranges = ["0–1", "0–2"]
elif bar == "MEDIO":
    ranges = ["1–2", "1–3"]
else:
    ranges = ["2–3", "2–4"]

step_box("STEP 2 — RANGE MULTIGOL CANDIDATI", [f"Range candidati: {', '.join(ranges)}"], "OK", "info")

# ---------- STEP 2B (coerenza split casa/trasferta) ----------
SPLIT_MIN_MATCHES = 6

# Baseline (stagione totale) già calcolata: bar / ranges
bar_base = bar
ranges_base = ranges

# Split coerente col match per la squadra valutata
if team_choice == "Casa":
    split_src = hs_home_gf
else:
    split_src = aw_away_gf

split_dist = _pct_dict_from_buckets(split_src["bucket_gf"].value_counts()) if len(split_src) else {k: 0.0 for k in ["G0","G1","G2","G3","G4+"]}
bar_split, s_g01, s_g12, s_g23 = compute_bar(split_dist)
ranges_split = ranges_for_bar(bar_split) if bar_split else []

st.subheader("STEP 2B — Verifica coerenza con split casa/trasferta (gol fatti)")
msg1 = f"Baseline (stagione): baricentro = **{bar_base}** | range = **{', '.join(ranges_base)}**"
msg2 = f"Split coerente (n={len(split_src)}): baricentro = **{bar_split if bar_split else 'NESSUNO'}** | range = **{', '.join(ranges_split) if ranges_split else '—'}**"
st.markdown(msg1 + "\n\n" + msg2)

split_ok = True
if len(split_src) < SPLIT_MIN_MATCHES or bar_split is None:
    st.info(f"Split con pochi match (min {SPLIT_MIN_MATCHES}) o baricentro non determinabile: lo uso solo come indicatore (non blocca).")
else:
    if bar_split == bar_base:
        st.success("✅ Coerente: lo split conferma il baricentro.")
    else:
        split_ok = False
        st.warning("⚠️ Non coerente: lo split sposta il baricentro → MG più rischioso. Consiglio: scegliere range più largo o ridurre stake.")


# ---------- STEP 3 ----------
g0c = opp_conc["G0"]
g3pc = opp_conc["G3"] + opp_conc["G4+"]

push_low = g0c >= 0.30
push_high = g3pc >= 0.20

step3_rows = [
    f"G0 subiti {opp_name} (ultime {min(N_LAST, len(away_away) if team_choice=='Casa' else len(home_home))}): {g0c:.0%} → {'↓ basso' if push_low else '—'}",
    f"G3+ subiti {opp_name}: {g3pc:.0%} → {'↑ alto' if push_high else '—'}",
]
conflict = push_low and push_high
if conflict:
    step3_rows.append("⚠️ Conflitto G0 alto + G3+ alto → MULTIGOL INSTABILE (passa a Under/BTTS)")
step_box("STEP 3 — GOL SUBITI AVVERSARI", step3_rows, "CONFLITTO" if conflict else "OK", "warning" if conflict else "info")

chosen_range = None
if not conflict:
    # elimina un range
    if push_low:
        chosen_range = ranges[0]  # quello più basso
    elif push_high:
        chosen_range = ranges[1]  # quello più alto
    else:
        # nessuna spinta: scegli quello più centrale (di solito il primo)
        chosen_range = ranges[0]

# ---------- STEP 4 (MG) ----------
def range_includes(range_str: str, k: str) -> bool:
    # k in ["G0","G1","G2","G3","G4+"]
    lo, hi = range_str.split("–")
    lo_i = int(lo)
    hi_i = int(hi)
    if k == "G4+":
        v = 4
    else:
        v = int(k[1])
    return lo_i <= v <= hi_i

def mg_cover(range_str: str, distd: dict) -> float:
    return sum(distd[k] for k in ["G0","G1","G2","G3","G4+"] if range_includes(range_str, k))

def excluded_strong_events(range_str: str, distd: dict, thr: float=0.30):
    out = []
    for k,p in distd.items():
        if (not range_includes(range_str, k)) and p >= thr:
            out.append((k,p))
    return out

mg_results = []
if not conflict and chosen_range:
    exc = excluded_strong_events(chosen_range, team_dist, 0.30)
    cover = mg_cover(chosen_range, team_dist)
    step4_rows = [f"Multigol candidato: {chosen_range}", f"Copertura (solo gol fatti {team_name}): {cover:.0%}"]
    if exc:
        step4_rows.append("❌ Esclude evento ≥30%: " + ", ".join([f"{k}({p:.0%})" for k,p in exc]))
        mg_ok = False
    else:
        mg_ok = True
        step4_rows.append("✔️ Non esclude eventi ≥30%")
    step_box("STEP 4 — CONTROLLO ESTREMI (MULTIGOL)", step4_rows, "VALIDO" if mg_ok else "SCARTATO", "success" if mg_ok else "error")
    if mg_ok:
        # label robustezza: 1 scenario perdita principale (fuori range) = instabile se somma eventi fuori range >=35%
        lose_ev = [(k,p) for k,p in team_dist.items() if not range_includes(chosen_range,k)]
        lose_sum = sum(p for _,p in lose_ev)
        label = "ROBUSTO" if lose_sum < 0.30 else ("NEUTRO" if lose_sum < 0.40 else "INSTABILE")

        # Se lo split (casa/trasferta) è NON coerente e abbiamo abbastanza match, abbasso di 1 livello la robustezza
        if (not split_ok) and (len(split_src) >= SPLIT_MIN_MATCHES):
            if label == "ROBUSTO":
                label = "NEUTRO"
            elif label == "NEUTRO":
                label = "INSTABILE"

        badge = "🟢" if label=="ROBUSTO" else ("🟡" if label=="NEUTRO" else "🔴")

        # Se è INSTABILE non lo proponiamo tra le scelte finali
        if label != "INSTABILE":
            mg_results.append({
                "name": f"MG {chosen_range} {team_name}",
                "label": label,
                "badge": badge,
                "lose_if": "Gol fuori range: " + ", ".join([k for k,p in lose_ev if p >= 0.10]) if lose_ev else "—",
                "kind": "MG",
                "cover": cover,
            })
else:
    st.info("Multigol non valutato (conflitto Step 3) → passo ai mercati alternativi.")

# ---------- STEP 4B (Under/Over/BTTS) ----------
alt_results = []

# Under decision (MATCH): proponilo solo se lo scenario è coerente per ENTRAMBE le squadre.
# Proxy: code 3+ basse sia per gol FATTI che per gol SUBITI (in contesto coerente).
team_for_g3p = team_for_ctx["G3"] + team_for_ctx["G4+"]
opp_for_g3p  = opp_for_ctx["G3"] + opp_for_ctx["G4+"]
team_conc_g3p = team_conc_ctx["G3"] + team_conc_ctx["G4+"]
opp_conc_g3p  = opp_conc_ctx["G3"] + opp_conc_ctx["G4+"]

under_ok = (team_for_g3p < 0.25) and (opp_for_g3p < 0.25) and (team_conc_g3p < 0.20) and (opp_conc_g3p < 0.20)

if under_ok:
    # scelta 2.5 vs 3.5: più "2.5" se la massa è su 0-1 (fatti+subiti)
    low_mass = (team_for_ctx["G0"] + team_for_ctx["G1"] + opp_for_ctx["G0"] + opp_for_ctx["G1"] +
                team_conc_ctx["G0"] + team_conc_ctx["G1"] + opp_conc_ctx["G0"] + opp_conc_ctx["G1"]) / 4.0
    under_choice = "Under 2.5" if low_mass >= 1.05 else "Under 3.5"  # 1.05 ~ media 0-1 >= 52.5%
    alt_results.append({
        "name": under_choice,
        "label": "ROBUSTO",
        "badge": "🟢",
        "lose_if": "3+ gol totali" if under_choice == "Under 2.5" else "4+ gol totali",
        "kind": "UNDER"
    })
# BTTS (MATCH) calibrato automaticamente.
# Stima P(team segna) e P(opp segna) combinando: G0 fatti (contesto) + G0 subiti avversario (contesto).
p_team_scores = 1.0 - ((team_for_ctx["G0"] + opp_conc_ctx["G0"]) / 2.0)
p_opp_scores  = 1.0 - ((opp_for_ctx["G0"] + team_conc_ctx["G0"]) / 2.0)
p_btts_yes = p_team_scores * p_opp_scores  # indipendenza (proxy)

# Proposta BTTS YES/NO solo se segnale netto
if (p_btts_yes >= 0.55) and (p_team_scores >= 0.65) and (p_opp_scores >= 0.65):
    alt_results.append({"name":"BTTS SI", "label":"NEUTRO", "badge":"🟡", "lose_if":"Almeno una a 0 gol", "kind":"BTTS"})
elif (p_btts_yes <= 0.45):
    alt_results.append({"name":"BTTS NO", "label":"NEUTRO", "badge":"🟡", "lose_if":"Entrambe segnano", "kind":"BTTS"})
# Over proxy: G2+ team e 2+ concessi opp
team_g2p = team_dist["G2"] + team_dist["G3"] + team_dist["G4+"]
opp_2p_conc = opp_conc["G2"] + opp_conc["G3"] + opp_conc["G4+"]
over_ok = (team_g2p >= 0.55) and (opp_2p_conc >= 0.45) and (team_dist["G0"] < 0.25)
if over_ok:
    over_choice = "Over 2.5" if team_g3p >= 0.25 else "Over 1.5 squadra"
    alt_results.append({"name":over_choice, "label":"NEUTRO", "badge":"🟡", "lose_if":"0-1 gol", "kind":"OVER"})

# ---------- STEP 5 (Trend blocker) ----------
delta = float(trend_row.get("Delta (ult6 - stag)", 0.0))
trend_rows = [f"Delta (ult6 - stag) {team_name}: {delta:+.2f}"]
trend_block = False
if delta >= 0.7:
    trend_rows.append("⚠️ Trend offensivo forte: evita Under / MG bassi")
elif delta <= -0.7:
    trend_rows.append("⚠️ Trend in calo: evita Over / MG alti")
else:
    trend_rows.append("✔️ Trend neutro: nessun blocco")
step_box("STEP 5 — TREND (ultime 6)", trend_rows, "BLOCCO" if abs(delta)>=0.7 else "OK", "warning" if abs(delta)>=0.7 else "success")

# ---------- OUTPUT FINALE ----------
all_results = mg_results + alt_results

# ordina per robustezza
order = {"ROBUSTO": 0, "NEUTRO": 1, "INSTABILE": 2}
all_results = sorted(all_results, key=lambda x: order.get(x.get("label","NEUTRO"), 1))
all_results = all_results[:3]

st.subheader("Esiti coerenti con i dati (ordinati per robustezza)")
if not all_results:
    st.warning("Nessun esito supera i filtri → NO BET")
else:
    for r in all_results:
        render_result(r)



st.divider()
st.subheader("Decisione finale (regola priorità)")

# Quote opzionali (se non le inserisci, la scelta è solo “numerica”)
c1, c2, c3 = st.columns(3)
with c1:
    q_mg = st.number_input("Quota MG (se presente)", min_value=1.01, max_value=50.0, value=float(quota), step=0.01)
with c2:
    q_under = st.number_input("Quota Under (se presente)", min_value=1.01, max_value=50.0, value=1.62, step=0.01)
with c3:
    q_btts = st.number_input("Quota BTTS (se presente)", min_value=1.01, max_value=50.0, value=1.62, step=0.01)

min_quota = st.number_input("Soglia minima quota", min_value=1.01, max_value=50.0, value=1.62, step=0.01)

# Priorità (semplice e coerente con la checklist):
# 1) Se Step 3 è in conflitto → niente MG (instabile) → prova Under/BTTS.
# 2) Se MG è ROBUSTO e copre bene (>=60%) → preferisci MG.
# 3) Se MG è NEUTRO ma Under è ROBUSTO → preferisci Under.
# 4) Altrimenti prendi il primo esito ROBUSTO/NEUTRO che supera la quota minima.
def pick_with_priority(results):
    # risultati già filtrati (niente INSTABILE)
    mg = [r for r in results if r.get("kind")=="MG"]
    under = [r for r in results if r.get("kind")=="UNDER"]
    btts = [r for r in results if r.get("kind")=="BTTS"]
    over = [r for r in results if r.get("kind")=="OVER"]

    # Se conflitto Step3, escludi MG
    if conflict:
        mg = []

    # Funzione quota per tipo
    def q_for(r):
        k = r.get("kind")
        if k == "MG":
            return q_mg
        if k == "UNDER":
            return q_under
        if k == "BTTS":
            return q_btts
        return float(quota)

    # 2) MG robusto e cover >=60%
    for r in mg:
        if r.get("label")=="ROBUSTO" and float(r.get("cover",0)) >= 0.60 and q_for(r) >= min_quota:
            return r, q_for(r), "MG ROBUSTO + copertura alta (≥60%)"

    # 3) Under robusto batte MG neutro
    for r in under:
        if r.get("label")=="ROBUSTO" and q_for(r) >= min_quota:
            # se c'è solo MG neutro o niente MG
            if not any(x.get("label")=="ROBUSTO" for x in mg):
                return r, q_for(r), "Under ROBUSTO (MG non robusto / assente)"

    # 4) fallback: primo che supera quota min in ordine robustezza già applicato
    for r in results:
        if q_for(r) >= min_quota:
            return r, q_for(r), "Miglior compromesso secondo robustezza"

    return None, None, "Nessun esito supera la soglia quota → NO BET"

picked, picked_q, why = pick_with_priority(all_results)

if picked is None:
    st.error("NO BET (con le tue soglie/quote)")
else:
    st.success(f"Scelta: **{picked['name']}**  | Quota usata: **{picked_q:.2f}**\n\nMotivo: {why}")

