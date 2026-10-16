# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import operator
import requests
import time

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
from statsmodels.tsa.stattools import acf, pacf 

# %%
FETCH_FROM_EDS = False
STORAGE_PATH = "../data/energidataservice/"


# %%
def get_eds_data(url):
    response = requests.get(url)
    result = response.json()
    if not "records" in result:
        raise ValueError(f"Could not find records in EnergiDataService response. Available: {result.keys()}")
    return pd.DataFrame(result["records"])

def read_csv(f):
    return pd.read_csv(f, date_format="ISO8601", parse_dates=["TimeUTC"])


# %%
def agg_resampled_mean(df, cols, freq="1D"):
    if "TimeUTC" not in cols:
        cols = ["TimeUTC"] + list(cols)
    return (
        df[cols]
        .set_index("TimeUTC")
        .resample(freq)
        .mean()
    )

def weighted_resampled_mean(df, cols, weights, freq="1D"):
    sum_cols = []
    df = df.copy()
    for c, w in zip(cols, weights):
        colname = c + "__times__" + w
        df[colname] = df[c] * df[w]
        sum_cols.append(colname)

    df = (
        df[["TimeUTC"] + weights + sum_cols]
        .set_index("TimeUTC")
        .resample(freq)
        .sum()
    )
    for nc, sc, wc in zip(cols, sum_cols, weights):
        df[nc] = df[sc] / df[wc]
    return df


# %% [markdown]
# ## Ancilliary services market - aFFR
# These are actually 2 markets:
# * **Capacity (reserves)** - what is bid before the day-ahead market. Dataset `AfrrReservesNordic`. Stakeholders that won a bid in the capacity market **must** create a matching bid in the energy market.
# * **Energy** - what is actually being used.
#
# ### Capacity market
# The demand and procured volume may not match exactly. Some of the demand can be covered by other TSOs and likewise, the procured volume may help meet other TSOs’ needs. This is especially true for DK2, which is part of the *Nordic Procurement from the capacity market is constant per hour of the day throughout each quarter. See a plan like this:
# https://en.energinet.dk/media/pxmkhs1w/market-information_afrr-procurement_q2-2026.pdf
#
# ### Energy market
# The dataset `AfrrEnergyActivation` shows each activation individually. However, this is a lot of data, and the dataset only keeps the last 3 months.
# A volume-weighted aFRR price per quarter hour is available from the `ImbalancePrice` dataset, which is used here instead.

# %%
_file_name = Path(STORAGE_PATH) / "afrr.csv"
if FETCH_FROM_EDS:
    afrr_reserves = get_eds_data("https://api.energidataservice.dk/dataset/AfrrReservesNordic?offset=0&start=2024-01-01T00:00&filter=%7B%22PriceArea%22:[%22DK2%22,%22DK1%22]%7D&sort=TimeUTC%20DESC")
    afrr_reserves = afrr_reserves.drop(columns=["TimeDK", "UpPriceDKK", "DownPriceDKK"])
    afrr_reserves["TimeUTC"] = pd.to_datetime(afrr_reserves["TimeUTC"])
    afrr_reserves.to_csv(_file_name, index=False)
else:
    afrr_reserves = read_csv(_file_name)
afrr_reserves.head()

# %%
_file_name = Path(STORAGE_PATH) / "imbalance_prices.csv"
if FETCH_FROM_EDS:
    imbalance_prices = get_eds_data("https://api.energidataservice.dk/dataset/ImbalancePrice?offset=0&start=2025-04-01T00:00&sort=TimeUTC%20DESC")
    imbalance_prices = imbalance_prices.drop(
        columns=["TimeDK", "ImbalancePriceDKK", "aFRRVWAUpDKK", "aFRRVWADownDKK", "mFRRMarginalPriceUpDKK", "mFRRMarginalPriceDownDKK"]
    )
    imbalance_prices["TimeUTC"] = pd.to_datetime(imbalance_prices["TimeUTC"])
    imbalance_prices.to_csv(_file_name, index=False)
else:
    imbalance_prices = read_csv(_file_name)

imbalance_prices = imbalance_prices.dropna()
imbalance_prices.head()


# %% [markdown]
# ### aFRR activations
# The aFRR **energy** market. Since this is a lot of data, we fetch it in daily batches. Pause a bit between two fetches, since EDS Api else throws an error. The data only goes 3 months back.

# %%
from collections.abc import Iterator
def afrr_energy_daily_urls(
    start: datetime,
    end: datetime,
    offset: timedelta = timedelta(days=1),
    base_url: str = "https://api.energidataservice.dk/dataset/AfrrEnergyActivation?offset=0"
) -> Iterator[str]:
    """Iterates over URLs to fetch data in batches."""
    current_start = start
    while current_start < end:
        current_end = current_start + offset
        if current_end > end:
            current_end = end
    
        url = (
            base_url
          + f"&start={current_start.strftime("%Y-%m-%dT%H:%M")}"  # OBS: ISO w/o seconds
          + f"&end={current_end.strftime("%Y-%m-%dT%H:%M")}"      # OBS: ISO w/o seconds
          + "&sort=TimeMsUTC%20DESC"
        )
        yield url
        current_start = current_end

def fetch_with_pauses(url_iterator: Iterator[str], sec_pause = 5):
    results = []
    for url in url_iterator:
        results.append(get_eds_data(url))
        time.sleep(sec_pause)
    
    return results


# %%
_file_name = Path(STORAGE_PATH) / "afrr_activations.parquet"
if FETCH_FROM_EDS:
    start = datetime(2026, 8, 1)
    end = datetime(2026, 10, 1)
    dfs = fetch_with_pauses(afrr_energy_daily_urls(start, end))
    afrr_activations = pd.concat(dfs)
    afrr_activations["TimeUTC"] = pd.to_datetime(afrr_activations["TimeMsUTC"], format="ISO8601")
    afrr_activations = afrr_activations.drop(columns=["TimeMsDK", "TimeMsUTC"])
    afrr_activations.to_parquet(_file_name)
else:
    afrr_activations = pd.read_parquet(_file_name)


# %% [markdown]
# Split by price area. Then we'll also create summary dataframes by quarter hour (the day-ahead and intraday frequency).

# %%
afrr_data = {}
for price_area, _df in afrr_activations.groupby("PriceArea"):
    afrr_data[price_area] = _df.drop(columns=["PriceArea"]).reset_index(drop=True)


# %% [markdown]
# For aFRR activations, we have to consider upregulations and downregulations separately, so they don't cancel out when summing.

# %%
def resample_afrr_activation(afrr_data, freq):
    """Resample aFRR activations, split by up- and downregulation."""
    result = {}
    for pa, df in afrr_data.items():
        result[pa] = {}
        for name, op in zip(["up", "down"], [operator.gt, operator.lt]):
            idx = op(df["aFRR_Activated"], 0)
            result[pa][name] = weighted_resampled_mean(
                df.loc[idx, :],
                ["aFRR_ActivatedEUR"],
                ["aFRR_Activated"],
                freq=freq,
            )

    return result
    


# %%
afrr_hourly = resample_afrr_activation(afrr_data, "1h")
afrr_quarterly = resample_afrr_activation(afrr_data, "15min")

# %%
afrr_quarterly["DK1"]["up"]

# %% [markdown]
# ## Day-ahead market
# ### Old ElSpotPrices API
# Data for the day-ahead market before 2025-10-01 comes from the old `ElSpotPrices` API with different column names.

# %%
_file_name = Path(STORAGE_PATH) / "spotprice.csv"
if FETCH_FROM_EDS:
    spotprice = get_eds_data("https://api.energidataservice.dk/dataset/Elspotprices?offset=0&start=2024-01-01T00:00&filter=%7B%22PriceArea%22:[%22DK1%22,%22DK2%22]%7D&sort=HourUTC%20DESC")
    spotprice = (
        spotprice.drop(columns=["HourDK", "SpotPriceDKK"])
        .rename(columns={"HourUTC": "TimeUTC", "SpotPriceEUR": "DayAheadPriceEUR"})
    )
    spotprice["TimeUTC"] = pd.to_datetime(spotprice["TimeUTC"])
    spotprice.to_csv(_file_name, index=False)
else:
    spotprice = read_csv(_file_name)
spotprice.head()

# %% [markdown]
# ### New DayAheadPrices API
# From 2025-10-01, we use the `DayAheadPrices` API. This one is in 15-minute blocks, so we need to aggregate.

# %%
_file_name = Path(STORAGE_PATH) / "day_ahead.csv"
if FETCH_FROM_EDS:
    day_ahead = get_eds_data("https://api.energidataservice.dk/dataset/DayAheadPrices?offset=0&start=2024-01-01T00:00&filter=%7B%22PriceArea%22:[%22DK1%22,%22DK2%22]%7D&sort=TimeUTC%20DESC")
    day_ahead = day_ahead.drop(columns=["TimeDK", "DayAheadPriceDKK"])
    day_ahead["TimeUTC"] = pd.to_datetime(day_ahead["TimeUTC"])
    day_ahead.to_csv(_file_name, index=False)
else:
    day_ahead = read_csv(_file_name)

_pa = []
for price_area, df in day_ahead.groupby("PriceArea"):
    df = df.set_index("TimeUTC").drop(columns="PriceArea").resample("1h").mean()
    df["PriceArea"] = price_area
    _pa.append(df.reset_index())

day_ahead = pd.concat(_pa)
day_ahead.head()

# %%
all_day_ahead = pd.concat([spotprice, day_ahead])
all_day_ahead.head()

# %% [markdown]
# ## Everything together, joined on time and price area

# %%
all_data = afrr_reserves.set_index(["PriceArea", "TimeUTC"]).join(all_day_ahead.set_index(["PriceArea", "TimeUTC"]))
all_data.head()

# %%
_data = {}
for price_area, _df in all_data.groupby(level="PriceArea"):
    _data[price_area] = _df.reset_index(level="PriceArea", drop=True)

dk1 = _data["DK1"].reset_index()
dk2 = _data["DK2"].reset_index()


# %%
def _filter_df(df, start=None, end=None):
    ts = df["TimeUTC"]
    start = pd.to_datetime(start) or ts.min()
    end = pd.to_datetime(end) or ts.max()

    return df.loc[(ts >= start) & (ts <= end), :]


def plot_power(df, start=None, end=None, legend_loc="best"):
    df = _filter_df(df, start, end)
    fig, ax = plt.subplots()
    
    for c, direction in zip(["C0", "C1"], ["Up", "Down"]):
        for ls, quantity in zip(["-", ":"], ["Demand", "Procured"]):
            col = direction + quantity + "MW"
            ax.plot(df["TimeUTC"], df[col], color=c, linestyle=ls, label=col)
    
    ax.legend(loc=legend_loc)
    ax.set_ylabel("Power [MW]")
    fig.autofmt_xdate()
    return fig, ax


# %%
def plot_power_prices(df, direction, start=None, end=None):
    direction = direction.title()
    if direction not in ["Up", "Down"]:
        raise ValueError("Allowed values for `direction` are 'Up', 'Down'")
    color_power = "C0"
    color_prices = "C5"
    
    df = _filter_df(df, start, end)

    fig, ax = plt.subplots()
    for ls, quantity in zip(["-", ":"], ["Demand", "Procured"]):
        col = direction + quantity + "MW"
        ax.plot(df["TimeUTC"], df[col], color=color_power, linestyle=ls, label=col)

    ax.set_ylabel("Power [MW]")
    ax.yaxis.label.set_color(color_power)
    ax.tick_params(axis="y", colors=color_power)

    ax2 = ax.twinx()
    y2_col = direction + "PriceEUR"
    ax2.plot(df["TimeUTC"], df[y2_col], color=color_prices, label="Price")
    ax2.set_ylabel("Price [EUR/MWh]")
    ax2.yaxis.label.set_color(color_prices)
    ax2.tick_params(axis="y", colors=color_prices)

    fig.autofmt_xdate()
    
    return fig, ax


# %%
fig, ax = plot_power(dk2, start="2026-06-01", end="2026-06-07")

# %%
fig, ax = plot_power_prices(dk1, direction="up", start="2026-06-01", end="2026-06-30")

# %%
plt_cols = ["UpPriceEUR", "DownPriceEUR"]
daily_prices = agg_resampled_mean(dk2, plt_cols)
daily_prices.iloc[730:].plot(y=plt_cols)

# %%
daily_prices_va = weighted_resampled_mean(dk2, ["UpPriceEUR", "DownPriceEUR"], ["UpProcuredMW", "DownProcuredMW"])
daily_prices_va.iloc[730:].plot(y=["UpPriceEUR", "DownPriceEUR"])

# %%
dk2.corr()

# %%
dk1.corr()

# %%
rxx = pacf(dk2["DayAheadPriceEUR"])
lags = np.arange(len(rxx))
fig, ax = plt.subplots()
ax.bar(lags, rxx)
ax.set(xlabel="lag [h]", ylabel="Part. autocorr")

# %%
_dk2 = dk2.loc[dk2.TimeUTC > pd.to_datetime("2025-06-01"), :]
cph_time = _dk2["TimeUTC"].dt.tz_localize("UTC").dt.tz_convert("Europe/Copenhagen")
_dk2["month"] = cph_time.dt.month
_dk2["hour"] = cph_time.dt.hour
dfs = {}
for _m, df in _dk2.loc[:, ["UpPriceEUR", "DownPriceEUR", "month", "hour"]].groupby("month"):
    dfs[_m] = df.drop(columns=["month"]).groupby("hour").agg(["mean", "std"])


# %%
fig, ax = plt.subplots()
N = len(dfs)
ax.set_prop_cycle(plt.cycler("color", plt.cm.twilight(np.linspace(0,1,N))))
for m, df in dfs.items():
    ax.plot(df.index, df[("UpPriceEUR", "mean")], label=str(m))

ax.set(xlabel="Hour of day", ylabel="Price [EUR]", title="Up-regulation per month and hour of day")
ax.legend()

# %%
fig, ax = plt.subplots()
ax.set_prop_cycle(plt.cycler("color", plt.cm.twilight(np.linspace(0,1,N))))
for m, df in dfs.items():
    ax.plot(df.index, df[("DownPriceEUR", "mean")], label=str(m))

ax.set(xlabel="Hour of day", ylabel="Price [EUR]", title="Down-regulation per month and hour of day")
ax.legend()

# %%
fig, ax = plt.subplots()

for pa, _data in afrr_hourly.items():
    for d, _df in _data.items():
        ax.plot(_df.index, _df["aFRR_ActivatedEUR"], label=f"{pa}: {d}")

ax.set(ylabel="aFRR price [EUR/MWh]")
ax.legend()
fig.autofmt_xdate()

# %% [markdown]
# ### aFRR: Correlation between prices and energy amounts

# %%
for pa, _data in afrr_hourly.items():
    for d, _df in _data.items():
        C = _df[["aFRR_Activated", "aFRR_ActivatedEUR"]].corr()
        print(pa + " " + d + ": " + f"{C.iloc[0, 1]:.3f}")

# %% [markdown]
# **Note**: For downregulation, the energy values are negative. The positive correlation therefore means that values closer to zero (little downregulation) carry higher prices that large downregulations!
# This is confirmed by the next plot. However, there are clearly two regimes for downregulation:
# * In hours with large downregulation, the prices are independent of the amount
# * In hours with very little downregulation, the prices increase with the amount

# %%
fig, ax = plt.subplots()

for pa, _data in afrr_quarterly.items():
    for d, _df in _data.items():
        # Used to only display downregulation below a threshold
        # if d == "up":
        #     continue
        # _df = _df.loc[_df["aFRR_Activated"] < -0.1e6, :]
        ax.scatter(_df["aFRR_Activated"], _df["aFRR_ActivatedEUR"], label=f"{pa}: {d}")

ax.set(xlabel="aFRR activated power [MW]", ylabel="aFRR price [EUR/MWh]")
ax.legend()

# %%
