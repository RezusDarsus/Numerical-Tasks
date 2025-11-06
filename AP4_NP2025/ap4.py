import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, Delaunay
from shapely.geometry import Polygon

df = pd.read_csv("kutaisi_taxi_stands.csv")

points = df[['lon', 'lat']].values


vor = Voronoi(points)

fig, ax = plt.subplots(figsize=(8, 8))

regions = vor.regions
vertices = vor.vertices

for region_index in vor.point_region:
    region = regions[region_index]
    if -1 not in region and len(region) > 0:
        polygon = [vertices[i] for i in region]
        poly = Polygon(polygon)
        if poly.is_valid and not poly.is_empty:
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.35, color=np.random.rand(3,))

for ridge in vor.ridge_vertices:
    if -1 not in ridge:
        ax.plot(vertices[ridge, 0], vertices[ridge, 1], 'k-', linewidth=1)

ax.scatter(points[:, 0], points[:, 1], color='red', s=45)

for i, row in df.iterrows():
    ax.text(row['lon'], row['lat'], row['name'], fontsize=9, color="black", ha='right')

ax.set_title("Voronoi Taxi Stand Service Zones in Kutaisi")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.show()


tri = Delaunay(points)

plt.figure(figsize=(8, 8))
plt.triplot(points[:, 0], points[:, 1], tri.simplices, color='blue')
plt.scatter(points[:, 0], points[:, 1], color='red', s=45)

for i, row in df.iterrows():
    plt.text(row['lon'], row['lat'], row['name'], fontsize=9, ha='right')

plt.title("Delaunay Taxi Stand Adjacency Network in Kutaisi")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
