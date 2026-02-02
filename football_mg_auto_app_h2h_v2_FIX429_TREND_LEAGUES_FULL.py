
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
    st.markdown(f"### {home_name} – Gol fatti (stagione, {len(hs)} match)")
    st.dataframe(dist_table(hs["bucket_gf"].value_counts(), len(hs)).style.format({"Percent":"{:.1%}"}),
                 use_container_width=True, hide_index=True)

with c2:
    st.markdown(f"### {away_name} – Gol subiti in trasferta (ultime {min(N_LAST, len(away_away))} partite)")
    st.dataframe(dist_table(away_away["bucket_ga"].value_counts(), len(away_away)).style.format({"Percent":"{:.1%}"}),
                 use_container_width=True, hide_index=True)

st.divider()

c3, c4 = st.columns(2)
with c3:
    st.markdown(f"### {away_name} – Gol fatti (stagione, {len(aw)} match)")
    st.dataframe(dist_table(aw["bucket_gf"].value_counts(), len(aw)).style.format({"Percent":"{:.1%}"}),
                 use_container_width=True, hide_index=True)

with c4:
    st.markdown(f"### {home_name} – Gol subiti in casa (ultime {min(N_LAST, len(home_home))} partite)")
    st.dataframe(dist_table(home_home["bucket_ga"].value_counts(), len(home_home)).style.format({"Percent":"{:.1%}"}),
                 use_container_width=True, hide_index=True)

st.divider()


st.divider()
st.subheader("Storico scontri diretti (H2H) – nella competizione selezionata")

st.markdown(
    """
Questa sezione cerca gli scontri diretti tra le due squadre **nella competizione selezionata**.

- Di default usa **solo la stagione corrente**.
- Se vuoi, puoi includere anche le stagioni precedenti (se l'API le rende disponibili).
- Mostra quante partite H2H sono state trovate e la distribuzione dei gol fatti da ciascuna squadra negli H2H.
"""
)

lookback_seasons = st.slider("Stagioni da includere (default ultime 5 se disponibili)", 0, 5, 5, 1)
max_h2h = st.slider("Max partite H2H da mostrare", 1, 20, 10, 1)

@st.cache_data(show_spinner=False, ttl=60*30)
def get_competition_matches_season(comp_code: str, season_year: int | None = None) -> pd.DataFrame:
    params = {}
    if season_year is not None:
        params["season"] = season_year
    data = api_get(f"/competitions/{comp_code}/matches", params=params if params else None)
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

def h2h_filter(df: pd.DataFrame, a_id: int, b_id: int) -> pd.DataFrame:
    if df.empty:
        return df
    cond1 = (df["home_id"] == a_id) & (df["away_id"] == b_id)
    cond2 = (df["home_id"] == b_id) & (df["away_id"] == a_id)
    out = df[cond1 | cond2].copy()
    out = out[out["status"] == "FINISHED"].dropna(subset=["home_ft", "away_ft"])
    out = out.sort_values("utcDate", ascending=False)
    return out

def team_goals_in_h2h(row: pd.Series, team_id: int) -> int | None:
    if row.get("status") != "FINISHED":
        return None
    if row.get("home_id") == team_id:
        return int(row.get("home_ft"))
    if row.get("away_id") == team_id:
        return int(row.get("away_ft"))
    return None

def bucket_0_4p_int(x: int) -> str:
    return f"G{x}" if x <= 3 else "G4+"

def h2h_dist(goals: pd.Series) -> pd.DataFrame:
    goals = goals.dropna().astype(int)
    total = len(goals)
    if total == 0:
        return pd.DataFrame({"Bucket":["G0","G1","G2","G3","G4+"], "Count":[0,0,0,0,0], "Percent":[0,0,0,0,0]})
    buckets = goals.apply(bucket_0_4p_int).value_counts()
    order = ["G0","G1","G2","G3","G4+"]
    rows = []
    for k in order:
        c = int(buckets.get(k, 0))
        rows.append({"Bucket": k, "Count": c, "Percent": c/total})
    return pd.DataFrame(rows)

# Prova a capire l'anno "stagione corrente" (approssimazione: anno corrente)
current_year = dt.date.today().year
season_years = [current_year] + [current_year - i for i in range(1, lookback_seasons + 1)]

h2h_frames = []
h2h_errors = []
for y in season_years:
    try:
        dfy = get_competition_matches_season(comp, season_year=y)
        h2h_frames.append(h2h_filter(dfy, home_id, away_id))
    except Exception as e:
        # Se la season non è supportata, ignoriamo e continuiamo
        h2h_errors.append(str(e))

h2h_all = pd.concat(h2h_frames, ignore_index=True) if h2h_frames else pd.DataFrame()

if h2h_all.empty:
    st.info("Nessuno scontro diretto trovato (o la stagione richiesta non è disponibile via API).")
    if h2h_errors:
        with st.expander("Dettagli errori (opzionale)"):
            for err in h2h_errors[:5]:
                st.code(err)
else:
    # Dedup per match_id
    h2h_all = h2h_all.drop_duplicates(subset=["match_id"]).sort_values("utcDate", ascending=False)
    st.write(f"Partite H2H trovate: **{len(h2h_all)}** (mostro le ultime {min(max_h2h, len(h2h_all))})")

    show_df = h2h_all.head(max_h2h).copy()
    show_df["Data"] = show_df["utcDate"].dt.tz_convert("Europe/Rome").dt.strftime("%d/%m/%Y") if "utcDate" in show_df else ""
    show_df["Risultato"] = show_df["home_ft"].astype(int).astype(str) + "-" + show_df["away_ft"].astype(int).astype(str)
    show_df = show_df[["Data","home_name","Risultato","away_name","matchday"]].rename(columns={"home_name":"Casa","away_name":"Trasferta","matchday":"MD"})
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    
    # Distribuzioni H2H con distinzione CASA/TRASFERTA per ciascuna squadra
    def h2h_goals_split(df: pd.DataFrame, team_id: int, side: str):
        if side == "home":
            mask = df["home_id"] == team_id
            goals_for = df.loc[mask, "home_ft"].astype(int)
            goals_against = df.loc[mask, "away_ft"].astype(int)
        else:
            mask = df["away_id"] == team_id
            goals_for = df.loc[mask, "away_ft"].astype(int)
            goals_against = df.loc[mask, "home_ft"].astype(int)
        return goals_for, goals_against

    def h2h_dist_table(goals: pd.Series) -> pd.DataFrame:
        total = len(goals)
        if total == 0:
            return pd.DataFrame({"Bucket":["G0","G1","G2","G3","G4+"], "Count":[0,0,0,0,0], "Percent":[0,0,0,0,0]})
        buckets = goals.apply(bucket_0_4p_int).value_counts()
        rows = []
        for k in ["G0","G1","G2","G3","G4+"]:
            c = int(buckets.get(k, 0))
            rows.append({"Bucket": k, "Count": c, "Percent": c/total})
        return pd.DataFrame(rows)

    cH, cA = st.columns(2)
    with cH:
        st.markdown(f"### {home_name} – H2H quando gioca in CASA")
        gf, ga = h2h_goals_split(h2h_all, home_id, "home")
        st.markdown("**Gol FATTI**")
        st.dataframe(h2h_dist_table(gf).style.format({"Percent":"{:.1%}"}), use_container_width=True, hide_index=True)
        st.markdown("**Gol SUBITI**")
        st.dataframe(h2h_dist_table(ga).style.format({"Percent":"{:.1%}"}), use_container_width=True, hide_index=True)
        st.caption(f"Match considerati: {len(gf)}")

    with cA:
        st.markdown(f"### {away_name} – H2H quando gioca in TRASFERTA")
        gf, ga = h2h_goals_split(h2h_all, away_id, "away")
        st.markdown("**Gol FATTI**")
        st.dataframe(h2h_dist_table(gf).style.format({"Percent":"{:.1%}"}), use_container_width=True, hide_index=True)
        st.markdown("**Gol SUBITI**")
        st.dataframe(h2h_dist_table(ga).style.format({"Percent":"{:.1%}"}), use_container_width=True, hide_index=True)
        st.caption(f"Match considerati: {len(gf)}")

st.caption(
    f"Produzione gol: stagione corrente (solo match FINISHED nella competizione). "
    f"Subiti: ultime {N_LAST} coerenti (casa per la squadra di casa, trasferta per la squadra ospite). "
    "Se l'API dà 429, è rate limit: attendi e riprova."
)
