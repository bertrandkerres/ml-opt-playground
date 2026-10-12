# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## ER Puzzle
# This notebook provides a solution for the [ER puzzle from InsideOpt](https://www.linkedin.com/pulse/er-meinolf-sellmann-gozbe/).
# It uses `jax` to do the calculations on the GPU (if available) and is therefore pretty fast, ~30 seconds to find a quite good solution.

# %%
import time

import jax
import jax.numpy as jnp
import optax

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
from jax import grad, jit, vmap
from jax import random

# %% [markdown]
# ### Data exploration and preparation

# %%
url = "https://cdn.shopify.com/s/files/1/0577/5602/4854/files/chicago_1m_residents.csv?v=1774701778"
df = pd.read_csv(url)
df.head()

# %%
limit_coords = (min(df["latitude"]), min(df["longitude"]), max(df["latitude"]), max(df["longitude"]))

def in_zone_1(lat, lon, coords):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    min_lat, min_lon, max_lat, max_lon = coords
    return (
        (lat >= 41.88)
      & (lat <= max_lat)
      & (lon >= min_lon)
      & (lon <= -87.65)
    )

def in_zone_2(lat, lon, coords):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    min_lat, min_lon, max_lat, max_lon = coords
    return (
        (lat >= min_lat)
      & (lat <= 41.88)
      & (lon >= -87.80)
      & (lon <= -87.62)
    )

def in_zone_3(lat, lon, coords):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    min_lat, min_lon, max_lat, max_lon = coords
    return (
        (lat >= 41.88)
      & (lat <= max_lat)
      & (lon >= -87.85)
      & (lon <= max_lon)
      & (lon <= -87.62 + 0.214 * (lat - 41.88))
    )

def in_zone_4(lat, lon, coords):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    min_lat, min_lon, max_lat, max_lon = coords
    return (
        (lat >= min_lat)
      & (lat <= 41.88)
      & (lon >= -87.82)
      & (lon <= max_lon)
      & (lon <= -87.52 + 0.769 * (lat - 41.75))
    )



# %%
means, stds = df.mean(), df.std()
df_norm = (df - means) / stds
df_norm.head()

# %%
df_norm.to_numpy()[:, 1]


# %%
# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1):
  key, _ = random.split(key)
  return scale * random.normal(key, (n, m))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def init_from_sample(locations, sizes, key):
    _, n = sizes
    key, _ = random.split(key)
    idx = random.choice(key, locations.shape[0], (n,))
    return jnp.array(locations[idx, :])

layer_sizes = [2, 80,]
params = init_network_params(layer_sizes, random.key(0))
for i, p in enumerate(params):
    print(f"Shape params[{i}]: {p.shape}")

sampled_params = [init_from_sample(df_norm.to_numpy(), (layer_sizes[0], layer_sizes[1]), random.key(0))]
for i, p in enumerate(sampled_params):
    print(f"Shape params[{i}]: {p.shape}")


# %%
scales = jnp.array([stds["latitude"], stds["longitude"]])
scales


# %% [markdown]
# ### Core Model
# The core model is similar to a 3-layer feedforward NN, where the inputs `(lat, lon)` are mapped to a distance to each ambulance in the middle layer.
# The output layer then just takes the `min` of all distances to the ambulances.
#
# That way, the weights of the middle layer are the coordinates of the ambulances, and minimizing some metric over all population clusters means we found a good solution with respect to that metric.

# %%
def coordinates_to_cost(a, b, scales):
    cost = jnp.array([222.2 * scales[0], 165.6 * scales[1]]) / 60  # In hours for better scaling
    return jnp.dot(jnp.abs(a - b), cost)


# %%
all_population_clusters = jnp.array(df_norm.to_numpy())
coordinates_to_cost(all_population_clusters[0, :], all_population_clusters[2, :], scales)

# %%
vectorized_cost = vmap(lambda a, b: coordinates_to_cost(a, b, scales), in_axes=(0, None))
vectorized_cost(params[0], all_population_clusters[0, :])

# %%
from jax.scipy.special import softmax

def single_predict(params, loc, T):
  # per-example predictions
  outputs = vectorized_cost(params[0], loc)
  # Note: This would give smoother gradients than the plain `min`
  # Temperature would need to be decreased towards 0
  # weights = softmax(-outputs / T)
  # return jnp.dot(weights, outputs)
  return jnp.min(outputs)
  


# %%
batched_predict = vmap(lambda p, x: single_predict(p, x, 0.01), in_axes=(None, 0))
B = batched_predict(params, all_population_clusters)
B


# %%
def loss(params, locs):
  min_distances = batched_predict(params, locs)
  return jnp.mean(min_distances) + 0.5 * jnp.quantile(min_distances, 0.95)


# %%
def fit(params, locs, step_size = 0.1, n_epochs: int = 10_000):
    
    @jit
    def update(params, locs):
        losses, grads = jax.value_and_grad(loss)(params, locs)
        params = [
            (w - step_size * dw)
            for (w, dw) in zip(params, grads)
        ]
        return params, losses

    for epoch in range(n_epochs):
        params, loss_value = update(params, locs)
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, loss: {loss_value}')     

    return params


# %%
def fit_with_optax(params, locs, optimizer, n_epochs: int = 10_000):
    opt_state = optimizer.init(params)

    loss_fn = lambda p: loss(p, locs)
    # value_and_grad = optax.value_and_grad_from_state(loss_fn)

    @jit
    def step(params, opt_state, locs):
        loss_value, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(
            grads, opt_state, params,
            value=loss_value,
            grad=grads,
            value_fn=loss_fn
        )
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    prev_loss = 0.0
    for epoch in range(n_epochs):
        params, opt_state, loss_value = step(params, opt_state, locs)
        if epoch % 100 == 0:
            if loss_value == prev_loss:
                print("No improvement for 100 epochs, stopping")
                break
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, loss: {loss_value}')
        prev_loss = loss_value

    return params, opt_state    


# %%
sched = optax.linear_schedule(
    init_value=0.1, end_value=0.001,
    transition_begin=10_000, transition_steps=100_000,
)
optimizer = optax.sgd(learning_rate = sched)
params_2, opt_state = fit_with_optax(sampled_params, all_population_clusters, optimizer, n_epochs = 50_000)


# %% [markdown]
# ### Evaluation

# %%
def eval_unscaled(params, scaled_locs, scales):
    vectorized_cost = vmap(lambda a, b: coordinates_to_cost(a, b, jnp.array([1, 1])), in_axes=(0, None))
    batched_cost = vmap(lambda a, b: vectorized_cost(a, b), in_axes=(None, 0))
    jit_batched_cost = jit(batched_cost)
    
    p = jnp.array(params[0])  # Raises exception if out of bounds
    scaled_locs = jnp.array(scaled_locs)
    
    unscaled_locs = jnp.vstack((scaled_locs[:, 0] * scales[0], scaled_locs[:, 1] * scales[1])).T
    unscaled_ambulances = jnp.vstack((p[:, 0] * scales[0], p[:, 1] * scales[1])).T

    return jit_batched_cost(unscaled_ambulances, unscaled_locs)



# %%
all_pairwise_cost = eval_unscaled(params_2, all_population_clusters, scales)

# %%
lats = np.asarray(params[0][:,0]) * stds["latitude"] + means["latitude"]
lons = np.asarray(params[0][:,1]) * stds["longitude"] + means["longitude"]
is_z1 = in_zone_1(lats, lons, limit_coords)
is_z2 = in_zone_2(lats, lons, limit_coords)
is_z3 = in_zone_3(lats, lons, limit_coords)
is_z4 = in_zone_4(lats, lons, limit_coords)

outside_any_zone = ~(is_z1 | is_z2 | is_z3 | is_z4)
print(f"No of ambulances placed where it's not allowed: {outside_any_zone.sum()}")

# %%
C = all_pairwise_cost[:, ~outside_any_zone]
mean_time = jnp.mean(jnp.min(C, axis=1)) * 60
q95_time = jnp.quantile(jnp.min(C, axis=1), 0.95) * 60
print(f"Mean: {mean_time:.2f}, q95: {q95_time:.2f}")

# %%
import geopandas as gpd
import contextily as cx


def visualize_stations(df_pop, lats, lons, title="New Ambulance Station Placements"):
    """
        Visualizes 50 stations clearly over the population density.
        """
    gdf_pop = gpd.GeoDataFrame(
        df_pop, geometry=gpd.points_from_xy(df_pop.longitude, df_pop.latitude), crs="EPSG:4326"
    ).to_crs(epsg=3857)

    # 3. Prepare Stations
    gdf_stations = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326"
    ).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(12, 15))

    # 4. Background Density
    hb = ax.hexbin(
        gdf_pop.geometry.x, gdf_pop.geometry.y,
        gridsize=300,
        cmap='viridis',  # Try 'magma', 'inferno', or 'plasma' for high contrast
        mincnt=1,
        alpha=0.9,  # Increased alpha for more "punch"
        zorder=2,
        bins='log'  # Use log scaling if you have a few very dense areas
    )
    # Add a colorbar so the colors actually mean something
    cb = fig.colorbar(hb, ax=ax, shrink=0.5, pad=0.01)

    """
    ax.hexbin(
        gdf_pop.geometry.x, gdf_pop.geometry.y,
        gridsize=300, cmap='YlGnBu', mincnt=1, alpha=0.3, zorder=1
    )
    """

    # 5. Stations
    # Note: Use 'jitter' or alpha if stations are overlapping
    gdf_stations.plot(
        ax=ax, color='red', marker='*', markersize=150,
        edgecolor='black', linewidth=0.5, label=f'Stations (Total: {len(lats)})',
        alpha=0.8, zorder=5
    )

    # Add Map
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zorder=0)

    ax.set_title("Ambulance Station Optimization", fontsize=16)
    ax.set_axis_off()
    plt.legend()
    plt.show()


# %%
visualize_stations(df, lats[~outside_any_zone], lons[~outside_any_zone])

# %%
