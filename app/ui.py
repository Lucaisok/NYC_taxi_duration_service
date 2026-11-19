"""Streamlit UI for the trip-duration API.

The UI only talks to the API (never to MLflow directly). Run with the API running:
    uv run streamlit run app/ui.py
"""

import os
import re
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
ZONES_CSV = Path(__file__).parent / "taxi_zone_lookup.csv"  # from NYC TLC
DEFAULT_PICKUP, DEFAULT_DROPOFF = 132, 230  # JFK Airport -> Times Square

st.set_page_config(page_title="NYC Taxi Trip Duration", page_icon="🚕")

# Custom styles for the parts the theme (.streamlit/config.toml) can't change
st.html("""
<style>
  /* hide Streamlit's own header bar (it covers the top of the page) and footer */
  [data-testid="stHeader"], footer {display: none;}
  .block-container, [data-testid="stMainBlockContainer"] {
         padding-top: 2.5rem; padding-bottom: 1.5rem; max-width: 760px;}

  /* the input card: its own background, lighter than the page */
  .st-key-trip_card {background: #33353B; border-radius: 18px; padding: 8px 6px;}

  /* centered loading spinner, shown while waiting for the API */
  .loader {display: flex; justify-content: center; padding: 28px 0;}
  .spinner {width: 44px; height: 44px; border-radius: 50%;
         border: 4px solid #4A4C53; border-top-color: #F5B700;
         animation: spin 0.8s linear infinite;}
  @keyframes spin {to {transform: rotate(360deg);}}

  /* yellow taxi banner: stands out on the dark page */
  .hero {background: #F5B700; border-radius: 20px; padding: 30px 32px 46px;
         position: relative; overflow: hidden; margin-bottom: 0.5rem;}
  .hero::after {content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 16px;
         background: conic-gradient(#111113 25%, #F5B700 0 50%, #111113 0 75%, #F5B700 0)
                     0 0 / 16px 16px;}  /* taxi checker stripe */
  .hero .badge {display: inline-block; background: #111113; color: #F5B700;
         font-weight: 700; font-size: 0.75rem; letter-spacing: 0.08em;
         padding: 4px 10px; border-radius: 999px; text-transform: uppercase;}
  .hero .title {color: #111113; font-size: 2.2rem; font-weight: 800;
         line-height: 1.15; margin: 14px 0 8px;}
  .hero .subtitle {color: #3D3300; font-size: 1rem; margin: 0;}

  .result {background: #33353B; border: 1px solid #4A4C53; border-left: 8px solid #F5B700;
         border-radius: 18px; padding: 22px 28px; margin-top: 0.5rem;}
  .result .label {color: #9C9890; font-size: 0.85rem; font-weight: 600;
         text-transform: uppercase; letter-spacing: 0.06em;}
  .result .value {font-size: 3.6rem; font-weight: 800; line-height: 1.05; color: #FFFFFF;}
  .result .value span {font-size: 1.4rem; font-weight: 600; color: #9C9890; margin-left: 6px;}
  .result .route {font-size: 1rem; color: #ECEAE4; margin-top: 10px;}
  .result .meta {font-size: 0.9rem; color: #9C9890; margin-top: 4px;}

  /* dark text on the yellow button for contrast */
  [data-testid^="stBaseButton-primary"] {color: #111113; font-weight: 700;}
  [data-testid^="stBaseButton-primary"] p {font-weight: 700; font-size: 1.05rem;}

  .footer {text-align: center; color: #75726B; font-size: 0.8rem; margin-top: 1.5rem;}
</style>
""")


# ---------- data ----------

@st.cache_data  # read the CSV once, not on every rerun
def load_zones() -> pd.DataFrame:
    zones = pd.read_csv(ZONES_CSV)
    # drop 264 "Unknown" and 265 "Outside of NYC" (pandas reads their "N/A" as missing)
    zones = zones.dropna(subset=["Borough", "Zone"])
    zones = zones[zones["Borough"] != "Unknown"]
    zones["label"] = zones["Zone"] + " · " + zones["Borough"]
    return zones.sort_values("label").set_index("LocationID")


@st.cache_data(ttl=60)  # ask the API at most once a minute
def fetch_model_version() -> str | None:
    # Errors are raised, not returned: st.cache_data doesn't cache exceptions,
    # so a failed call (e.g. API still starting) is retried on the next rerun
    response = requests.get(f"{API_URL}/", timeout=5)
    response.raise_for_status()
    return response.json().get("model_version")


def load_model_version() -> str | None:
    try:
        return fetch_model_version()
    except requests.RequestException:
        return None  # the UI still works without this info


def predict_minutes(pickup_id: int, dropoff_id: int, pickup_at: datetime) -> int:
    response = requests.post(
        f"{API_URL}/predict",
        json={
            "pulocationid": pickup_id,
            "dolocationid": dropoff_id,
            "tpep_pickup_datetime": pickup_at.isoformat(),
        },
        timeout=10,
    )
    response.raise_for_status()
    # the API answers e.g. {"duration": "15 minutes"}
    duration = response.json()["duration"]
    match = re.match(r"\d+", duration)
    if match is None:
        raise ValueError(f"Unexpected duration format from the API: {duration!r}")
    return int(match.group())


zones = load_zones()
zone_ids: list[int] = zones.index.tolist()
# plain dicts (id -> text) are simpler for the dropdowns than DataFrame lookups
zone_labels: dict[int, str] = dict(zip(zone_ids, zones["label"].astype(str)))
zone_names: dict[int, str] = dict(zip(zone_ids, zones["Zone"].astype(str)))
model_version = load_model_version()


# ---------- page ----------

st.html("""
<div class="hero">
  <div class="badge">🚕 NYC Yellow Cab</div>
  <div class="title">How long will my taxi ride take?</div>
  <p class="subtitle">Pick your zones and pickup time. The model predicts the trip
     duration from 3.3 million real trips.</p>
</div>
""")

# No st.form: every input change reruns the script, so an old result disappears
# as soon as the user picks a new zone, date or time.
with st.container(border=True, key="trip_card"):  # key -> CSS class .st-key-trip_card
    st.markdown("##### Route")
    col_from, col_to = st.columns(2)
    pickup_id = col_from.selectbox(
        "Pickup zone",
        zone_ids,
        index=zone_ids.index(DEFAULT_PICKUP),
        format_func=zone_labels.__getitem__,
    )
    dropoff_id = col_to.selectbox(
        "Drop-off zone",
        zone_ids,
        index=zone_ids.index(DEFAULT_DROPOFF),
        format_func=zone_labels.__getitem__,
    )

    st.markdown("##### When")
    col_date, col_time = st.columns(2)
    pickup_date = col_date.date_input("Pickup date", value=date.today(), format="DD/MM/YYYY")
    pickup_time = col_time.time_input("Pickup time", value=time(8, 30), step=timedelta(minutes=15))

    clicked = st.button("Estimate duration", type="primary", width="stretch")

# One slot for spinner -> result: the spinner replaces the old result immediately
result_slot = st.empty()

if clicked:
    result_slot.html('<div class="loader"><div class="spinner"></div></div>')
    pickup_at = datetime.combine(pickup_date, pickup_time)
    try:
        minutes = predict_minutes(pickup_id, dropoff_id, pickup_at)
    except requests.ConnectionError:
        result_slot.error(f"Can't reach the prediction API at {API_URL}. Is it running?")
    except (requests.RequestException, ValueError) as error:
        result_slot.error(f"The API returned an error: {error}")
    else:
        arrival = pickup_at + timedelta(minutes=minutes)
        result_slot.html(f"""
        <div class="result">
          <div class="label">Estimated trip duration</div>
          <div class="value">{minutes}<span>min</span></div>
          <div class="route">{zone_names[pickup_id]} → {zone_names[dropoff_id]}</div>
          <div class="meta">Arrive around {arrival:%H:%M} on {arrival:%a %d %b}</div>
        </div>
        """)

model_text = (
    f"Model v{model_version} (champion in the MLflow registry)"
    if model_version
    else "Model info unavailable (API not reachable)"
)
st.html(f'<div class="footer">{model_text} · Trained on NYC TLC trips, January 2025</div>')
