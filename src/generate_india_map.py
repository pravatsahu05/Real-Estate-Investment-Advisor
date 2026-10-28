import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from matplotlib.path import Path as MplPath
from pathlib import Path

# Outline coordinates of India (Lat, Lon)
india_boundary_pts = [
    # Kanyakumari and West Coast
    (8.08, 77.55), (8.88, 76.60), (9.98, 76.28), (11.25, 75.78), (12.50, 74.98),
    (14.00, 74.45), (15.40, 73.80), (16.98, 73.30), (18.95, 72.80), (19.80, 72.75),
    (20.50, 72.85), (21.15, 72.60), (20.80, 71.40), (20.70, 70.90), (21.50, 69.60),
    (22.25, 68.95), (22.80, 69.80), (23.20, 68.50), (23.80, 68.80), (24.20, 70.80),
    (24.60, 71.10),
    # Rajasthan / Gujarat border to Punjab
    (25.50, 70.50), (26.80, 70.20), (27.80, 71.50), (28.80, 72.80), (30.00, 73.80),
    (31.20, 74.50), (32.20, 75.20),
    # Jammu & Kashmir / Ladakh
    (32.80, 74.80), (33.80, 74.30), (34.50, 74.00), (35.50, 75.00), (36.00, 76.50),
    (35.50, 77.80), (34.80, 79.00), (33.50, 79.20), (32.50, 78.80), (31.50, 78.50),
    (31.00, 77.80), (30.50, 78.20), (30.00, 80.20),
    # Along Nepal border (Uttarakhand, UP, Bihar)
    (28.80, 80.10), (28.20, 81.20), (27.50, 82.50), (27.00, 84.00), (26.80, 85.50),
    (26.40, 87.20),
    # Sikkim & Bhutan border
    (27.20, 88.10), (28.00, 88.60), (27.70, 88.90), (27.00, 88.90), (26.80, 89.80),
    (27.30, 91.80),
    # Arunachal Pradesh
    (27.80, 92.50), (28.40, 93.80), (29.00, 95.00), (28.50, 96.50), (28.00, 97.00),
    (27.20, 96.50),
    # Nagaland, Manipur, Mizoram
    (26.50, 95.20), (25.50, 94.50), (24.50, 94.00), (23.50, 93.50), (22.20, 93.00),
    (21.80, 92.80), (22.80, 92.30), (23.80, 91.80), (24.20, 92.00),
    # Meghalaya / Assam
    (25.10, 91.50), (25.20, 90.00), (25.80, 89.80), (26.20, 89.90),
    # West Bengal / Bangladesh border to Sundarbans
    (25.20, 88.80), (24.00, 88.50), (23.00, 88.90), (21.75, 89.00), (21.60, 88.00),
    (21.50, 87.00),
    # Odisha Coast
    (20.50, 86.80), (19.80, 85.80), (19.20, 84.90),
    # Andhra Pradesh Coast
    (18.20, 83.90), (17.70, 83.30), (16.50, 82.20), (15.80, 80.80), (14.50, 80.20),
    # Tamil Nadu Coast
    (13.40, 80.25), (13.08, 80.27), (12.00, 79.80), (10.80, 79.85), (10.30, 79.80),
    (9.30, 79.00), (9.00, 78.20), (8.40, 77.80), (8.08, 77.55)
]

cities = {
    'DELHI NCR': {'lat': 28.6139, 'lon': 77.2090, 'psf': 'INR 10.5k/sqft', 'cagr': '10.8%', 'col': '#00F2FE'},
    'MUMBAI': {'lat': 18.9500, 'lon': 72.8200, 'psf': 'INR 13.2k/sqft', 'cagr': '9.8%', 'col': '#F5C518'},
    'PUNE': {'lat': 18.5204, 'lon': 73.8567, 'psf': 'INR 7.2k/sqft', 'cagr': '11.2%', 'col': '#A855F7'},
    'BENGALURU': {'lat': 12.9716, 'lon': 77.5946, 'psf': 'INR 8.4k/sqft', 'cagr': '12.4%', 'col': '#00F2FE'},
    'HYDERABAD': {'lat': 17.3850, 'lon': 78.4867, 'psf': 'INR 6.8k/sqft', 'cagr': '13.1%', 'col': '#22C55E'},
    'CHENNAI': {'lat': 13.0827, 'lon': 80.2707, 'psf': 'INR 7.1k/sqft', 'cagr': '9.4%', 'col': '#38BDF8'},
    'KOLKATA': {'lat': 22.5726, 'lon': 88.3639, 'psf': 'INR 5.4k/sqft', 'cagr': '8.2%', 'col': '#EC4899'},
    'AHMEDABAD': {'lat': 23.0225, 'lon': 72.5714, 'psf': 'INR 4.9k/sqft', 'cagr': '10.2%', 'col': '#EAB308'}
}

# Corridors between cities
corridors = [
    ('MUMBAI', 'PUNE'),
    ('MUMBAI', 'AHMEDABAD'),
    ('MUMBAI', 'DELHI NCR'),
    ('DELHI NCR', 'KOLKATA'),
    ('BENGALURU', 'CHENNAI'),
    ('BENGALURU', 'HYDERABAD'),
    ('HYDERABAD', 'CHENNAI'),
    ('PUNE', 'HYDERABAD'),
    ('AHMEDABAD', 'DELHI NCR')
]

fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
fig.patch.set_facecolor('#070A18')
ax.set_facecolor('#070A18')

# Draw cyber grid
for lon in np.arange(64, 100, 4):
    ax.axvline(lon, color='#00F2FE', alpha=0.07, lw=0.6, linestyle=':')
for lat in np.arange(5, 40, 4):
    ax.axhline(lat, color='#00F2FE', alpha=0.07, lw=0.6, linestyle=':')

# Draw concentric radar rings around center of India
center_lat, center_lon = 21.5, 78.5
for r in [4, 8, 12, 16, 20]:
    circle = plt.Circle((center_lon, center_lat), r, color='#00F2FE', fill=False, lw=0.6, alpha=0.08, linestyle='--')
    ax.add_patch(circle)

# Plot India Boundary with glow
lats = [p[0] for p in india_boundary_pts]
lons = [p[1] for p in india_boundary_pts]

# Fill mainland with subtle dark neon cyan tint
ax.fill(lons, lats, color='#0B132B', alpha=0.92)

# Multi-layer glow lines for boundary
for width, alpha in [(10, 0.08), (5, 0.22), (2.0, 0.85)]:
    ax.plot(lons, lats, color='#00F2FE', lw=width, alpha=alpha, solid_capstyle='round')

# Fill internal dot matrix
lon_grid = np.linspace(69, 95, 75)
lat_grid = np.linspace(8.5, 35.5, 75)
india_path = MplPath(list(zip(lons, lats)))

pts_inside_x = []
pts_inside_y = []
for x in lon_grid:
    for y in lat_grid:
        if india_path.contains_point((x, y)):
            pts_inside_x.append(x)
            pts_inside_y.append(y)

ax.scatter(pts_inside_x, pts_inside_y, color='#00F2FE', s=2.2, alpha=0.22, marker='.')

# Draw Corridors with glowing cyan/purple curved paths
for c1, c2 in corridors:
    p1 = cities[c1]
    p2 = cities[c2]
    # Draw curved path
    mx = (p1['lon'] + p2['lon']) / 2 + (p1['lat'] - p2['lat']) * 0.07
    my = (p1['lat'] + p2['lat']) / 2 + (p2['lon'] - p1['lon']) * 0.07
    t = np.linspace(0, 1, 35)
    curve_x = (1-t)**2 * p1['lon'] + 2*(1-t)*t * mx + t**2 * p2['lon']
    curve_y = (1-t)**2 * p1['lat'] + 2*(1-t)*t * my + t**2 * p2['lat']
    ax.plot(curve_x, curve_y, color='#00F2FE', lw=1.4, alpha=0.45, linestyle='--')

# Plot Metros with glowing markers and callouts
for name, d in cities.items():
    # Outer radar ring
    circ1 = plt.Circle((d['lon'], d['lat']), 0.9, color=d['col'], fill=False, lw=1.4, alpha=0.55)
    circ2 = plt.Circle((d['lon'], d['lat']), 1.6, color=d['col'], fill=False, lw=0.7, alpha=0.30, linestyle=':')
    ax.add_patch(circ1)
    ax.add_patch(circ2)
    
    # Bright center dot
    ax.scatter([d['lon']], [d['lat']], color='#FFFFFF', s=55, zorder=10, edgecolors=d['col'], linewidths=3.0)
    
    # Text label box
    offset_x = 1.0 if d['lon'] < 81.5 else -1.0
    ha = 'left' if d['lon'] < 81.5 else 'right'
    if name == 'MUMBAI':
        offset_x, ha = -1.2, 'right'
    elif name == 'PUNE':
        offset_x, ha = 1.2, 'left'
    elif name == 'AHMEDABAD':
        offset_x, ha = -1.2, 'right'
    elif name == 'DELHI NCR':
        offset_x, ha = 1.2, 'left'
        
    label_txt = f"{name}\n{d['psf']} • {d['cagr']} 5Y"
    ax.text(d['lon'] + offset_x, d['lat'], label_txt, color='#FFFFFF', fontsize=9.5, fontweight='bold',
            ha=ha, va='center', family='sans-serif',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#0B1124', edgecolor=d['col'], alpha=0.92, lw=1.4))

# Ambient Telemetry overlays
ax.text(65.5, 36.8, '[ REAL ESTATE INVESTMENT ADVISOR • ALL-INDIA LOCATION GRID ]',
        color='#00F2FE', fontsize=13, fontweight='bold', family='monospace', alpha=0.9)

ax.text(65.5, 35.6, '12,000 Verified Listings | 8 Metropolitan Clusters | 200+ Micro-Markets',
        color='#8A99AD', fontsize=9.5, family='monospace', alpha=0.75)

ax.text(65.5, 7.2, '● ML ENGINE ONLINE: HistGradientBoosting (95.2% AUC) | Ridge Regressor (R2 0.994)\n● CORRIDORS: Mumbai-Pune-Ahmedabad, Bangalore-Hyderabad-Chennai, Delhi NCR-Kolkata',
        color='#8A99AD', fontsize=8.5, family='monospace', alpha=0.7)

ax.set_xlim(63, 99)
ax.set_ylim(6, 38.5)
ax.set_aspect('equal')
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

out_path = Path('assets/india_holographic_map.png')
out_path.parent.mkdir(exist_ok=True)
plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', pad_inches=0)
plt.close()

print('Successfully generated holographic India map:', out_path.resolve())
