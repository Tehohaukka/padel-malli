"""
Premier Padel Vetomalli
- Elo-rating (pelaajatasoinen, päivitetään kronologisesti)
- Serve- ja return-tilastot (viimeinen 20 ottelua per pelaaja)
- Logistinen regressio yhdistää featuret ML%-arvioksi
"""

import difflib
import os
import re
import warnings
from collections import defaultdict
from datetime import datetime as _dt

import numpy as np
import pandas as pd
import requests as _requests
import streamlit as st
from bs4 import BeautifulSoup as _BS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(_DIR, "padel_data.csv")
HISTORY_PATH = os.path.join(_DIR, "vetohistoria.csv")
K_ELO   = 32
INIT_ELO = 1500.0
WINDOW   = 20


# ── Apufunktiot ───────────────────────────────────────────────────────────────

def parse_frac(s: str) -> float | None:
    """Palauttaa osuuden muodosta '74% 20/27' -> 0.7407, tai None."""
    m = re.search(r"(\d+)/(\d+)", str(s))
    if m:
        n, d = int(m.group(1)), int(m.group(2))
        return n / d if d > 0 else None
    return None


def fmt_pct(v: float | None) -> str:
    return f"{v * 100:.1f} %" if v is not None else "—"


def fmt_odds(v: float) -> str:
    """Todennäköisyys -> desimaalikertoimeksi (fair odds)."""
    return f"{1 / v:.2f}" if v > 0 else "—"


# ── Datan lataus ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df[df["status"] == "Completed"].copy()
    df["winner"] = pd.to_numeric(df["winner"], errors="coerce")
    df = df.dropna(subset=["winner"])
    df["winner"] = df["winner"].astype(int)

    for col in ["t1_p1", "t1_p2", "t2_p1", "t2_p2"]:
        df[col] = df[col].astype(str).str.strip()

    # Parsitaan päästatistiikat
    for side in ["t1", "t2"]:
        df[f"srv_{side}"] = df[f"match_total_serve_points_won_{side}"].apply(parse_frac)
        df[f"ret_{side}"] = df[f"match_total_return_points_won_{side}"].apply(parse_frac)

    df = df.sort_values("match_id").reset_index(drop=True)
    return df


# ── Bradley-Terry ─────────────────────────────────────────────────────────────

def _bt_players_for_gender(
    rows: list,
    alpha: float,
    max_iter: int,
    tol: float,
) -> dict:
    """
    Bradley-Terry at PLAYER level via MM algorithm.
    Each row is (p1a, p1b, p2a, p2b, won) — four players, pair 1 vs pair 2.
    A win for pair 1 counts as a win for both p1a and p1b against both p2a and p2b.
    Returns {player_name -> log_strength}, normalized so geometric mean = 0.
    """
    wins: dict = defaultdict(float)
    n_mat: dict = defaultdict(lambda: defaultdict(float))

    for p1a, p1b, p2a, p2b, won in rows:
        team1 = [p1a, p1b]
        team2 = [p2a, p2b]
        for pi in team1:
            for pj in team2:
                wins[pi] += float(won)
                wins[pj] += float(not won)
                n_mat[pi][pj] += 1.0
                n_mat[pj][pi] += 1.0

    players = sorted(wins.keys())
    NEU = "__neutral__"
    for p in players:
        wins[p] += alpha
        wins[NEU] += alpha
        n_mat[p][NEU] += alpha
        n_mat[NEU][p] += alpha

    all_keys = players + [NEU]
    s = {p: 1.0 for p in all_keys}

    for _ in range(max_iter):
        ns = {}
        for i in all_keys:
            w_i = wins[i]
            denom = sum(
                n_mat[i][j] / (s[i] + s[j])
                for j in n_mat[i] if n_mat[i][j] > 0
            )
            ns[i] = (w_i / denom) if denom > 0 and w_i > 0 else 1e-9
        log_mean = float(np.mean([np.log(v) for v in ns.values() if v > 1e-10]))
        scale = np.exp(log_mean)
        ns = {p: v / scale for p, v in ns.items()}
        delta = max(abs(ns[p] - s[p]) for p in all_keys)
        s = ns
        if delta < tol:
            break

    return {p: float(np.log(s[p])) for p in players}


def _compute_bt(
    df: pd.DataFrame,
    alpha: float = 2.0,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> dict:
    """
    Bradley-Terry player strengths via MM algorithm, per the book's MLE approach.
    P(pair_i beats pair_j) = 1 / (1 + exp(-(mean_log_s_i - mean_log_s_j)))
    Computed separately for Men and Women to avoid cross-gender contamination.
    All matches (including qualifiers) used — qualifiers connect through shared players
    and create the full strength calibration chain.
    alpha: Laplace-smoothing regularization against neutral baseline.
    Returns {player_name -> log_strength}, normalized so geometric mean = 0.
    Unknown players default to log_strength=0 at prediction time.
    """
    men_rows: list = []
    women_rows: list = []

    for _, r in df.iterrows():
        cat = str(r.get("category", "")).strip()
        won = (r["winner"] == 1)
        row = (r["t1_p1"], r["t1_p2"], r["t2_p1"], r["t2_p2"], won)
        if cat.startswith("Women"):
            women_rows.append(row)
        else:
            men_rows.append(row)

    bt_men   = _bt_players_for_gender(men_rows,   alpha, max_iter, tol)
    bt_women = _bt_players_for_gender(women_rows, alpha, max_iter, tol)
    return {**bt_men, **bt_women}


def _pair_bt(player_bt: dict, pair: list[str]) -> float:
    """Pair BT log-strength = mean of player log-strengths (geometric mean in linear scale)."""
    vals = [player_bt.get(p, 0.0) for p in pair]
    return float(np.mean(vals))


# ── Elo + tilastojen laskenta ─────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_all(df: pd.DataFrame):
    """
    Käy ottelut kronologisesti läpi. Joka ottelulle kirjataan
    PRE-match Elo ja rolling-tilastot (featureiksi mallille).
    Bradley-Terry lasketaan koko datasta (parikohtainen vahvuus).
    Ottelun jälkeen päivitetään Elo ja tilastohistoria.
    """
    bt_log = _compute_bt(df)

    elo: dict[str, float] = defaultdict(lambda: INIT_ELO)
    p_srv: dict[str, list[float]] = defaultdict(list)
    p_ret: dict[str, list[float]] = defaultdict(list)

    feat_rows: list[dict] = []

    for _, r in df.iterrows():
        t1 = [r["t1_p1"], r["t1_p2"]]
        t2 = [r["t2_p1"], r["t2_p2"]]

        def team_elo(pl: list[str]) -> float:
            return float(np.mean([elo[p] for p in pl]))

        def team_stat(pl: list[str], hist: dict) -> float:
            vals = [np.mean(hist[p][-WINDOW:]) for p in pl if hist[p]]
            return float(np.mean(vals)) if vals else float("nan")

        e1, e2 = team_elo(t1), team_elo(t2)
        s1, s2 = team_stat(t1, p_srv), team_stat(t2, p_srv)
        rv1, rv2 = team_stat(t1, p_ret), team_stat(t2, p_ret)
        bt_diff = _pair_bt(bt_log, t1) - _pair_bt(bt_log, t2)

        feat_rows.append(
            dict(
                elo_diff=e1 - e2,
                bt_diff=bt_diff,
                srv_diff=0.0 if (np.isnan(s1) or np.isnan(s2)) else s1 - s2,
                ret_diff=0.0 if (np.isnan(rv1) or np.isnan(rv2)) else rv1 - rv2,
                winner=r["winner"],
            )
        )

        # Päivitetään Elo
        E1 = 1.0 / (1.0 + 10.0 ** ((e2 - e1) / 400.0))
        S1 = 1.0 if r["winner"] == 1 else 0.0
        delta = K_ELO * (S1 - E1)
        for p in t1:
            elo[p] += delta
        for p in t2:
            elo[p] -= delta

        # Päivitetään rolling-tilastot
        for side, players in [("t1", t1), ("t2", t2)]:
            sv = r[f"srv_{side}"]
            rv = r[f"ret_{side}"]
            for p in players:
                if pd.notna(sv):
                    p_srv[p].append(float(sv))
                if pd.notna(rv):
                    p_ret[p].append(float(rv))

    feat_df = pd.DataFrame(feat_rows)
    return feat_df, dict(elo), dict(p_srv), dict(p_ret), bt_log


# ── Mallin koulutus ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def train_model(feat_df: pd.DataFrame):
    X = feat_df[["elo_diff", "bt_diff", "srv_diff", "ret_diff"]].fillna(0.0).values
    y = (feat_df["winner"] == 1).astype(int).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(random_state=42, max_iter=500, C=1.0)
    clf.fit(Xs, y)

    acc = float((clf.predict(Xs) == y).mean())
    return clf, scaler, acc


# ── Ennuste ───────────────────────────────────────────────────────────────────

def predict(
    pair1: list[str],
    pair2: list[str],
    elo: dict,
    p_srv: dict,
    p_ret: dict,
    bt_log: dict,
    clf: LogisticRegression,
    scaler: StandardScaler,
) -> tuple[float, float, float, dict]:

    def pe(pl: list[str]) -> float:
        return float(np.mean([elo.get(p, INIT_ELO) for p in pl]))

    def ps(pl: list[str], hist: dict) -> float | None:
        vals = [float(np.mean(hist[p][-WINDOW:])) for p in pl if hist.get(p)]
        return float(np.mean(vals)) if vals else None

    e1, e2 = pe(pair1), pe(pair2)
    s1, s2 = ps(pair1, p_srv), ps(pair2, p_srv)
    r1, r2 = ps(pair1, p_ret), ps(pair2, p_ret)

    bt1 = _pair_bt(bt_log, pair1)
    bt2 = _pair_bt(bt_log, pair2)
    bt_diff = bt1 - bt2
    bt_prob = float(1.0 / (1.0 + np.exp(-bt_diff)))

    elo_diff = e1 - e2
    srv_diff = (s1 - s2) if (s1 is not None and s2 is not None) else 0.0
    ret_diff = (r1 - r2) if (r1 is not None and r2 is not None) else 0.0

    X = np.array([[elo_diff, bt_diff, srv_diff, ret_diff]])
    ensemble_prob = float(clf.predict_proba(scaler.transform(X))[0][1])
    elo_prob = 1.0 / (1.0 + 10.0 ** ((e2 - e1) / 400.0))

    details = dict(
        e1=e1, e2=e2, s1=s1, s2=s2, r1=r1, r2=r2,
        bt1=bt1, bt2=bt2,
        elo_diff=elo_diff, bt_diff=bt_diff, srv_diff=srv_diff, ret_diff=ret_diff,
    )
    return ensemble_prob, elo_prob, bt_prob, details


# ── H2H-haku ─────────────────────────────────────────────────────────────────

def head_to_head(df: pd.DataFrame, pair1: list[str], pair2: list[str]) -> tuple[int, int]:
    s1, s2 = set(pair1), set(pair2)

    def row_teams(r) -> tuple[set, set]:
        return {r.t1_p1, r.t1_p2}, {r.t2_p1, r.t2_p2}

    w1 = w2 = 0
    for r in df.itertuples():
        ta, tb = row_teams(r)
        if ta == s1 and tb == s2:
            if r.winner == 1:
                w1 += 1
            else:
                w2 += 1
        elif ta == s2 and tb == s1:
            if r.winner == 1:
                w2 += 1
            else:
                w1 += 1
    return w1, w2


# ── Vetohistoria – tallennus ─────────────────────────────────────────────────

_HIST_COLS = ["id", "pvm", "ottelu", "pari", "kerroin", "panos", "tulos", "voitto"]


def load_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_PATH):
        return pd.DataFrame(columns=_HIST_COLS)
    df = pd.read_csv(HISTORY_PATH, dtype={"tulos": str})
    df["tulos"] = df["tulos"].fillna("")
    return df


def save_history(df: pd.DataFrame) -> None:
    df.to_csv(HISTORY_PATH, index=False)


def _profit(kerroin: float, panos: float, tulos: str) -> float:
    if tulos == "W":
        return round((kerroin - 1) * panos, 2)
    if tulos == "L":
        return round(-panos, 2)
    return 0.0


def add_bet(ottelu: str, pari: str, kerroin: float, panos: float, tulos: str = "") -> None:
    df = load_history()
    new_id = int(df["id"].max()) + 1 if not df.empty else 1
    row = pd.DataFrame([{
        "id":      new_id,
        "pvm":     pd.Timestamp.now().strftime("%Y-%m-%d"),
        "ottelu":  ottelu,
        "pari":    pari,
        "kerroin": kerroin,
        "panos":   panos,
        "tulos":   tulos,
        "voitto":  _profit(kerroin, panos, tulos),
    }])
    save_history(pd.concat([df, row], ignore_index=True))


def update_result(bet_id: int, tulos: str) -> None:
    df = load_history()
    mask = df["id"] == bet_id
    df.loc[mask, "tulos"]  = tulos
    df.loc[mask, "voitto"] = df[mask].apply(
        lambda r: _profit(r["kerroin"], r["panos"], tulos), axis=1
    )
    save_history(df)


def delete_bet(bet_id: int) -> None:
    df = load_history()
    save_history(df[df["id"] != bet_id].reset_index(drop=True))


# ── Otteluohjelma: Premier Padel API ────────────────────────────────────────

_PP_BASE = "https://premierpadel.com"
_PP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": _PP_BASE + "/en/tournaments",
    "Accept":  "application/json, */*",
}
_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _pp_post(path: str, **form_data) -> dict:
    r = _requests.post(
        _PP_BASE + path,
        data={k: str(v) for k, v in form_data.items()},
        headers=_PP_HEADERS,
        timeout=12,
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_upcoming_tournaments() -> list[dict]:
    """Hakee tulevat / käynnissä olevat turnaukset (nykyinen + 2 seuraavaa kuukautta)."""
    now = _dt.now()
    found: list[dict] = []
    seen: set[str] = set()
    for delta in range(3):
        total_month = now.month - 1 + delta
        year  = now.year + total_month // 12
        month = _MONTH_NAMES[total_month % 12]
        try:
            result = _pp_post(
                "/premierpadel/api/beforeauth/getfanapptournaments",
                page="", pagesize="-1", month_name=month, year=str(year),
            )
            for t in result.get("data", []):
                slug   = t.get("slug", "")
                status = t.get("tournaments_type", "")
                if not slug or slug in seen:
                    continue
                if status not in ("Upcoming", "Live", "Active", "Current"):
                    continue
                seen.add(slug)
                found.append({
                    "name":   t.get("full_name", slug),
                    "slug":   slug,
                    "id":     str(t.get("tournaments_id", "")),
                    "start":  t.get("start_date", ""),
                    "end":    t.get("end_date", ""),
                    "status": status,
                })
        except Exception:
            continue
    return found


@st.cache_data(ttl=900, show_spinner=False)
def fetch_tournament_matches(tournament_id: str, slug: str) -> list[dict]:
    """
    Hakee turnauksen otteluohjelman kahdella API-kutsulla:
      1. gettournamentsdate(slug)  →  turnauksen päivät
      2. gettournamentsmatchlistnew(tournaments_id, date, draw_type)  ×  päivät × Men/Women
    Palauttaa tulevat + meneillään olevat ottelut.
    """
    # Vaihe 1: turnauksen päivät
    try:
        date_resp = _pp_post(
            "/premierpadel/api/beforeauth/gettournamentsdate",
            slug=slug, lang="en",
        )
        all_dates = [d["date"] for d in date_resp.get("data", [])]
    except Exception:
        return []

    if not all_dates:
        return []

    today = _dt.now().strftime("%Y-%m-%d")
    fetch_dates = [d for d in all_dates if d >= today] or all_dates[-2:]

    # Vaihe 2: ottelut per päivä per sukupuoli
    matches: list[dict] = []
    seen_ids: set[str] = set()

    for date in fetch_dates:
        for gender in ("Men", "Women"):
            try:
                resp = _pp_post(
                    "/premierpadel/api/beforeauth/gettournamentsmatchlistnew",
                    tournaments_id=tournament_id,
                    date=date,
                    draw_type=gender,
                    lang="en",
                )
                bucket = resp.get("data", {})
                is_live = str(resp.get("is_live_match", "")).lower() == "yes"
                raw = (
                    bucket.get("main_draw",    []) +
                    bucket.get("qualify_draw", []) +
                    bucket.get("live",         []) +
                    bucket.get("upcoming",     [])
                )
                for m in raw:
                    mid = str(m.get("tournaments_match_id", ""))
                    if not mid or mid in seen_ids:
                        continue
                    seen_ids.add(mid)
                    # Ohita jo valmistuneet
                    if m.get("status") == "F" and str(m.get("winner_id", "")) in ("1", "2"):
                        continue
                    matches.append({
                        "match_id":   mid,
                        "round":      m.get("round_name", ""),
                        "date":       m.get("date", date),
                        "start_time": m.get("start_time", ""),
                        "court":      m.get("court_name", ""),
                        "t1_p1":      m.get("team1_player_name", ""),
                        "t1_p2":      m.get("team1_partner_name", ""),
                        "t2_p1":      m.get("team2_player_name", ""),
                        "t2_p2":      m.get("team2_partner_player_name", ""),
                        "live":       is_live or m.get("status") == "L",
                    })
            except Exception:
                continue

    return matches


def _fuzzy_player(name: str, pool: list[str]) -> str:
    """Löytää lähimmän pelaajan nimen tietokannasta."""
    if not name or not pool:
        return pool[0] if pool else ""
    name_l = name.strip().lower()
    # 1) Tarkka substring
    for p in pool:
        if name_l in p.lower() or p.lower() in name_l:
            return p
    # 2) Sukunimi-haku (viimeinen sana)
    last = name_l.split()[-1] if name_l.split() else name_l
    for p in pool:
        if last in p.lower():
            return p
    # 3) Fuzzy
    hits = difflib.get_close_matches(name, pool, n=1, cutoff=0.3)
    return hits[0] if hits else pool[0]


# ── Pelaajahaku ──────────────────────────────────────────────────────────────

def player_picker(label: str, key: str, pool: list[str], exclude: set[str]) -> str:
    """
    text_input + selectbox -yhdistelmä. Kirjoittamalla suodatetaan ensin
    osumahakuna, sitten fuzzy-hakuna (difflib). pool on valmiiksi järjestetty.
    """
    available = [p for p in pool if p not in exclude]

    query = st.sidebar.text_input(label, key=f"{key}_q", placeholder="Kirjoita nimi…")

    if query.strip():
        q = query.strip().lower()
        # 1) Tarkka substring-osuus
        candidates = [p for p in available if q in p.lower()]
        # 2) Fuzzy-varmuuskopio jos ei osumia
        if not candidates:
            candidates = difflib.get_close_matches(query, available, n=12, cutoff=0.35)
        if not candidates:
            st.sidebar.caption("Ei tuloksia – näytetään kaikki")
            candidates = available
    else:
        candidates = available

    if not candidates:
        st.sidebar.warning("Ei vapaita pelaajia")
        return pool[0] if pool else ""

    return st.sidebar.selectbox(
        "​",  # zero-width space → otsikko piilotettu visuaalisesti
        candidates,
        key=f"{key}_sel",
        label_visibility="collapsed",
    )


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Premier Padel Vetomalli", page_icon="🎾", layout="wide")

# Ladataan data (aina tarvitaan)
with st.spinner("Ladataan data ja lasketaan ratingit…"):
    df = load_data()
    feat_df, elo, p_srv, p_ret, bt_log = compute_all(df)
    clf, scaler, acc = train_model(feat_df)

match_counts: dict[str, int] = defaultdict(int)
for _, r in df.iterrows():
    for col in ["t1_p1", "t1_p2", "t2_p1", "t2_p2"]:
        match_counts[r[col]] += 1
all_players = sorted(match_counts.keys())
elo_sorted  = sorted(all_players, key=lambda p: elo.get(p, INIT_ELO), reverse=True)

# ── Sivunavigointi ────────────────────────────────────────────────────────────

_PAGES    = ["📅 Otteluohjelma", "🎾 Vetomalli", "📋 Vetohistoria"]
_PAGE_KEY = {"otteluohjelma": 0, "vetomalli": 1, "vetohistoria": 2}
_PAGE_QP  = {0: "otteluohjelma", 1: "vetomalli", 2: "vetohistoria"}

_qp       = st.query_params.get("page", "vetomalli")
_page_idx = _PAGE_KEY.get(_qp, 1)

page = st.sidebar.radio(
    "Sivu",
    _PAGES,
    index=_page_idx,
    label_visibility="collapsed",
)
# Pidä query param URL synkronoituna manuaaliseen valintaan
st.query_params["page"] = _PAGE_QP[_PAGES.index(page)]
st.sidebar.divider()

# ═══════════════════════════════════════════════════════════════════════════════
#  SIVU 1 — OTTELUOHJELMA
# ═══════════════════════════════════════════════════════════════════════════════

if page == "📅 Otteluohjelma":

    st.title("📅 Otteluohjelma")
    st.caption("Tulevat Premier Padel -ottelut · Päivitetään 30 min välein")

    if st.button("🔄 Päivitä"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Haetaan turnaukset…"):
        tournaments = fetch_upcoming_tournaments()

    if not tournaments:
        st.info(
            "Ei tulevia turnauksia juuri nyt. "
            "Premierpadel.com saattaa olla tilapäisesti poissa tai turnauksia ei ole "
            "lähiviikkoina."
        )
        st.stop()

    for t in tournaments:
        st.subheader(f"🏆 {t['name']}")
        st.caption(f"{t['start']} – {t['end']}  ·  {t['status']}")

        with st.spinner(f"Haetaan {t['name']} ottelut…"):
            matches = fetch_tournament_matches(t["id"], t["slug"])

        if not matches:
            st.info("Otteluohjelma ei ole vielä saatavilla tälle turnaukselle.")
            st.divider()
            continue

        # Ryhmittele kategorian mukaan
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for m in matches:
            by_cat[m.get("category") or "—"].append(m)

        for cat, cat_matches in by_cat.items():
            st.markdown(f"**{cat}**")
            for m in cat_matches:
                t1_str = f"{m['t1_p1']} / {m['t1_p2']}" if m["t1_p2"] else m["t1_p1"]
                t2_str = f"{m['t2_p1']} / {m['t2_p2']}" if m["t2_p2"] else m["t2_p1"]
                live_badge = " 🔴 LIVE" if m.get("live") else ""

                col_match, col_info, col_btn = st.columns([5, 3, 2])
                col_match.markdown(f"**{t1_str}** vs **{t2_str}**{live_badge}")

                time_str = m.get("start_time", "")
                court_str = m.get("court", "")
                meta_parts = [p for p in [m.get("date",""), time_str, court_str] if p]
                col_info.caption(" · ".join(meta_parts) or "—")

                if col_btn.button("🎾 Analysoi", key=f"sched_{m['match_id']}"):
                    st.session_state["p1a_q"] = _fuzzy_player(m["t1_p1"], elo_sorted)
                    st.session_state["p1b_q"] = _fuzzy_player(m["t1_p2"], elo_sorted)
                    st.session_state["p2a_q"] = _fuzzy_player(m["t2_p1"], elo_sorted)
                    st.session_state["p2b_q"] = _fuzzy_player(m["t2_p2"], elo_sorted)
                    st.query_params["page"] = "vetomalli"
                    st.rerun()

        st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
#  SIVU 2 — VETOMALLI
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🎾 Vetomalli":

    st.title("🎾 Premier Padel — Vetomalli")

    # Sivupalkki: pelaajat + kerroin
    st.sidebar.title("Valitse parit")
    st.sidebar.caption("Kirjoita nimi hakukenttään — ehdottaa lähimpiä nimiä")

    st.sidebar.subheader("Pari 1")
    p1_a = player_picker("Pelaaja A", "p1a", elo_sorted, set())
    p1_b = player_picker("Pelaaja B", "p1b", elo_sorted, {p1_a})

    st.sidebar.subheader("Pari 2")
    p2_a = player_picker("Pelaaja C", "p2a", elo_sorted, {p1_a, p1_b})
    p2_b = player_picker("Pelaaja D", "p2b", elo_sorted, {p1_a, p1_b, p2_a})

    st.sidebar.divider()
    bookie_odds = st.sidebar.number_input(
        "Kirjanpitäjän kerroin (Pari 1)", min_value=1.01, max_value=50.0,
        value=2.00, step=0.05,
        help="Syötä vedonlyöntikertoimen arvo, niin näet onko se arvokas",
    )
    st.sidebar.divider()
    st.sidebar.caption(f"📊 {len(df):,} ottelua · {len(all_players):,} pelaajaa")
    st.sidebar.caption(f"Mallin tarkkuus: {acc * 100:.1f} %")

    # Lasketaan ennuste
    pair1 = [p1_a, p1_b]
    pair2 = [p2_a, p2_b]
    ensemble_prob, elo_prob, bt_prob, d = predict(
        pair1, pair2, elo, p_srv, p_ret, bt_log, clf, scaler
    )
    h2h_w1, h2h_w2 = head_to_head(df, pair1, pair2)
    total_h2h = h2h_w1 + h2h_w2
    name1 = f"{p1_a} / {p1_b}"
    name2 = f"{p2_a} / {p2_b}"

    # BT-vahvuus rankina (exp(log_strength), suurempi = vahvempi)
    bt_sorted_pairs = sorted(bt_log.items(), key=lambda x: x[1], reverse=True)
    bt_rank = {p: i + 1 for i, (p, _) in enumerate(bt_sorted_pairs)}
    n_players = len(bt_log)

    # Tiimikortit
    st.divider()
    col_t1, col_vs, col_t2 = st.columns([5, 1, 5])
    with col_t1:
        st.subheader(name1)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Elo (tiimi)", f"{d['e1']:.0f}")
        avg_rank1 = int(round(np.mean([bt_rank.get(p, n_players) for p in pair1])))
        c2.metric("BT-vahvuus", f"{np.exp(d['bt1']):.2f}",
                  help="Bradley-Terry vahvuusparametri. >1 = vahvempi kuin keskiarvo.")
        c2.caption(f"Rank #{avg_rank1}/{n_players} (keskim.)")
        c3.metric("Serve %", fmt_pct(d["s1"]))
        c4.metric("Return %", fmt_pct(d["r1"]))
    with col_vs:
        st.markdown("<br><br><h3 style='text-align:center'>vs</h3>", unsafe_allow_html=True)
        if total_h2h:
            st.markdown(
                f"<p style='text-align:center;font-size:1.1em'>H2H<br><b>{h2h_w1}–{h2h_w2}</b></p>",
                unsafe_allow_html=True,
            )
    with col_t2:
        st.subheader(name2)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Elo (tiimi)", f"{d['e2']:.0f}")
        avg_rank2 = int(round(np.mean([bt_rank.get(p, n_players) for p in pair2])))
        c2.metric("BT-vahvuus", f"{np.exp(d['bt2']):.2f}",
                  help="Bradley-Terry vahvuusparametri. >1 = vahvempi kuin keskiarvo.")
        c2.caption(f"Rank #{avg_rank2}/{n_players} (keskim.)")
        c3.metric("Serve %", fmt_pct(d["s2"]))
        c4.metric("Return %", fmt_pct(d["r2"]))

    st.divider()

    # Todennäköisyydet
    st.subheader("Voittotodennäköisyydet")
    prob_col1, prob_col2 = st.columns(2)

    fair  = 1.0 / ensemble_prob      if ensemble_prob > 0 else 999.0
    fair2 = 1.0 / (1 - ensemble_prob) if ensemble_prob < 1 else 999.0
    edge  = (ensemble_prob * bookie_odds - 1.0) * 100

    with prob_col1:
        st.markdown(f"**{name1}**")
        st.progress(ensemble_prob, text=f"Ensemble: {ensemble_prob * 100:.1f} %")
        m1, m2, m3 = st.columns(3)
        m1.metric("Ensemble", f"{ensemble_prob * 100:.1f} %")
        m2.metric("Bradley-Terry", f"{bt_prob * 100:.1f} %")
        m3.metric("Elo",     f"{elo_prob * 100:.1f} %")
        st.markdown("---")
        e1c, e2c = st.columns(2)
        e1c.metric("Fair kerroin", f"{fair:.2f}")
        e2c.metric(
            "Arvo (Edge)", f"{edge:+.1f} %",
            delta="✅ Arvokas" if edge > 0 else "❌ Ei arvokas",
            delta_color="normal" if edge > 0 else "inverse",
        )
    with prob_col2:
        st.markdown(f"**{name2}**")
        st.progress(1 - ensemble_prob, text=f"Ensemble: {(1 - ensemble_prob) * 100:.1f} %")
        m1, m2, m3 = st.columns(3)
        m1.metric("Ensemble", f"{(1 - ensemble_prob) * 100:.1f} %")
        m2.metric("Bradley-Terry", f"{(1 - bt_prob) * 100:.1f} %")
        m3.metric("Elo",     f"{(1 - elo_prob) * 100:.1f} %")
        st.markdown("---")
        st.metric("Fair kerroin", f"{fair2:.2f}")

    # Tarkempi analyysi
    with st.expander("🔍 Tarkempi analyysi"):
        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Elo-ero (P1 − P2)", f"{d['elo_diff']:+.0f}")
        fc2.metric("BT log-ero",
                   f"{d['bt_diff']:+.3f}",
                   help="log(λ₁) − log(λ₂). Positiivinen = Pari 1 vahvempi BT:n mukaan.")
        fc3.metric("Serve-ero",  f"{d['srv_diff'] * 100:+.1f} pp" if d["srv_diff"] else "Ei dataa")
        fc4.metric("Return-ero", f"{d['ret_diff'] * 100:+.1f} pp" if d["ret_diff"] else "Ei dataa")

        st.markdown("#### Mallin kertoimet (standardoidut)")
        coefs = dict(zip(["Elo-ero", "BT log-ero", "Serve-ero", "Return-ero"], clf.coef_[0]))
        st.dataframe(
            pd.DataFrame([{"Feature": k, "Kerroin": f"{v:.3f}"} for k, v in coefs.items()]),
            hide_index=True, use_container_width=False,
        )

        st.markdown("#### Pelaajien tilastot (viim. 20 ottelua)")
        player_rows = []
        for pair, pname in [(pair1, name1), (pair2, name2)]:
            for p in pair:
                srv_hist = p_srv.get(p, [])
                ret_hist = p_ret.get(p, [])
                player_rows.append({
                    "Pelaaja": p, "Pari": pname,
                    "Elo": f"{elo.get(p, INIT_ELO):.0f}",
                    "Serve %":  fmt_pct(np.mean(srv_hist[-WINDOW:]) if srv_hist else None),
                    "Return %": fmt_pct(np.mean(ret_hist[-WINDOW:]) if ret_hist else None),
                    "Otteluita": match_counts.get(p, 0),
                })
        st.dataframe(pd.DataFrame(player_rows), hide_index=True, use_container_width=True)

        if total_h2h:
            st.markdown(f"#### Head-to-Head: {h2h_w1}–{h2h_w2} ({name1} vs {name2})")
            h2h_rows = []
            s1s, s2s = set(pair1), set(pair2)
            for r in df.itertuples():
                ta, tb = {r.t1_p1, r.t1_p2}, {r.t2_p1, r.t2_p2}
                if (ta == s1s and tb == s2s) or (ta == s2s and tb == s1s):
                    won1 = (ta == s1s and r.winner == 1) or (ta == s2s and r.winner == 2)
                    h2h_rows.append({
                        "Turnaus": r.tournament_name, "Vuosi": r.tournament_year,
                        "Kierros": r.category, "Voittaja": name1 if won1 else name2,
                    })
            st.dataframe(pd.DataFrame(h2h_rows), hide_index=True, use_container_width=True)

    # Pikalisäys vetohistoriaan
    st.divider()
    with st.expander("➕ Tallenna tämä veto historiaan"):
        with st.form("quick_add_bet", clear_on_submit=True):
            qc1, qc2, qc3, qc4 = st.columns([3, 2, 1, 1])
            q_ottelu = qc1.text_input("Ottelu", value=f"{name1} vs {name2}")
            q_pari   = qc2.text_input("Vetattu pari", value=name1)
            q_kerr   = qc3.number_input("Kerroin", min_value=1.01, value=round(fair, 2), step=0.05)
            q_panos  = qc4.number_input("Panos (u)", min_value=0.1, value=1.0, step=0.5)
            q_tulos  = st.radio("Tulos", ["Avoin", "W", "L"], horizontal=True)
            if st.form_submit_button("💾 Tallenna veto"):
                add_bet(q_ottelu, q_pari, q_kerr, q_panos,
                        "" if q_tulos == "Avoin" else q_tulos)
                st.success("Veto tallennettu!")

# ═══════════════════════════════════════════════════════════════════════════════
#  SIVU 3 — VETOHISTORIA
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📋 Vetohistoria":

    st.title("📋 Vetohistoria")

    df_h = load_history()

    # ── Tilastot ──────────────────────────────────────────────────────────────

    settled = df_h[df_h["tulos"].isin(["W", "L"])] if not df_h.empty else pd.DataFrame()
    n_total    = len(df_h)
    n_settled  = len(settled)
    n_wins     = int((settled["tulos"] == "W").sum()) if n_settled else 0
    hit_pct    = n_wins / n_settled * 100  if n_settled else 0.0
    tot_profit = float(settled["voitto"].sum()) if n_settled else 0.0
    tot_staked = float(settled["panos"].sum())  if n_settled else 0.0
    roi        = tot_profit / tot_staked * 100   if tot_staked else 0.0

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.metric("Vetoja yhteensä",  n_total)
    sc2.metric("Selvitetty",       f"{n_settled}  (avoimia: {n_total - n_settled})")
    sc3.metric("Osumis%",          f"{hit_pct:.1f} %")
    sc4.metric("ROI%",             f"{roi:+.1f} %",
               delta_color="normal" if roi >= 0 else "inverse")
    sc5.metric("Voitto / tappio",  f"{tot_profit:+.2f} u",
               delta_color="normal" if tot_profit >= 0 else "inverse")

    st.divider()

    # ── Lisää uusi veto ───────────────────────────────────────────────────────

    with st.expander("➕ Lisää uusi veto", expanded=n_total == 0):
        with st.form("new_bet", clear_on_submit=True):
            nc1, nc2 = st.columns(2)
            n_ottelu = nc1.text_input("Ottelu (esim. Coello/Tapia vs Lebron/Galan)")
            n_pari   = nc2.text_input("Vetattu pari")
            nc3, nc4, nc5 = st.columns(3)
            n_kerr  = nc3.number_input("Kerroin", min_value=1.01, max_value=100.0, value=2.00, step=0.05)
            n_panos = nc4.number_input("Panos (yksikköä)", min_value=0.1, value=1.0, step=0.5)
            n_tulos = nc5.selectbox("Tulos", ["Avoin", "W", "L"])
            if st.form_submit_button("💾 Lisää veto") and n_ottelu.strip():
                add_bet(n_ottelu.strip(), n_pari.strip(), n_kerr, n_panos,
                        "" if n_tulos == "Avoin" else n_tulos)
                st.rerun()

    # ── Avoimet vedot ─────────────────────────────────────────────────────────

    open_bets = df_h[df_h["tulos"] == ""] if not df_h.empty else pd.DataFrame()

    if not open_bets.empty:
        st.subheader(f"Avoimet vedot ({len(open_bets)})")
        hdr = st.columns([3, 2, 1, 1, 1, 1, 0.6])
        for lbl, col in zip(["Ottelu", "Pari", "Kerroin", "Panos", "", "", ""], hdr):
            col.markdown(f"**{lbl}**")

        for _, row in open_bets.iterrows():
            bid = int(row["id"])
            rc = st.columns([3, 2, 1, 1, 1, 1, 0.6])
            rc[0].write(row["ottelu"])
            rc[1].write(row["pari"])
            rc[2].write(f"{row['kerroin']:.2f}")
            rc[3].write(f"{row['panos']:.1f} u")
            if rc[4].button("✅ W", key=f"w_{bid}"):
                update_result(bid, "W")
                st.rerun()
            if rc[5].button("❌ L", key=f"l_{bid}"):
                update_result(bid, "L")
                st.rerun()
            if rc[6].button("🗑", key=f"d_{bid}"):
                delete_bet(bid)
                st.rerun()

        st.divider()

    # ── Kaikki selvitetyt vedot ────────────────────────────────────────────────

    if n_settled:
        st.subheader("Selvitetyt vedot")
        disp = (
            settled[["pvm", "ottelu", "pari", "kerroin", "panos", "tulos", "voitto"]]
            .sort_values("pvm", ascending=False)
            .rename(columns={
                "pvm": "Päivä", "ottelu": "Ottelu", "pari": "Pari",
                "kerroin": "Kerroin", "panos": "Panos (u)",
                "tulos": "Tulos", "voitto": "Voitto (u)",
            })
        )
        st.dataframe(
            disp.style.map(
                lambda v: "color: #2ecc71" if v == "W" else ("color: #e74c3c" if v == "L" else ""),
                subset=["Tulos"],
            ).map(
                lambda v: "color: #2ecc71" if isinstance(v, float) and v > 0
                          else ("color: #e74c3c" if isinstance(v, float) and v < 0 else ""),
                subset=["Voitto (u)"],
            ),
            hide_index=True,
            use_container_width=True,
        )

    elif n_total == 0:
        st.info("Ei vielä yhtään vetoa. Lisää ensimmäinen veto yllä olevalla lomakkeella.")
