"""
Synthetic Dataset Generator for India Housing Prices
Strictly matches the schema and requirements outlined in the Real Estate Investment Advisor specification.
"""

import numpy as np
import pandas as pd
import random
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_housing_dataset(n_samples: int = 12000, output_path: str = "data/india_housing_prices.csv"):
    current_year = 2026
    
    city_state_map = {
        "Mumbai": {
            "state": "Maharashtra",
            "base_psf": 18500,
            "growth_rate_mean": 0.082,
            "localities": {
                "Bandra West": 2.4,
                "Andheri West": 1.4,
                "Powai": 1.3,
                "Worli": 2.8,
                "Borivali": 0.95,
                "Thane West": 0.75,
                "Navi Mumbai": 0.65,
                "Goregaon East": 1.15
            }
        },
        "Bengaluru": {
            "state": "Karnataka",
            "base_psf": 9200,
            "growth_rate_mean": 0.105,
            "localities": {
                "Indiranagar": 1.9,
                "Koramangala": 1.8,
                "Whitefield": 1.05,
                "HSR Layout": 1.35,
                "Electronic City": 0.75,
                "Hebbal": 1.15,
                "Marathahalli": 0.95,
                "Bellandur": 1.25
            }
        },
        "Delhi NCR": {
            "state": "Delhi NCR",
            "base_psf": 9800,
            "growth_rate_mean": 0.088,
            "localities": {
                "Golf Course Road (Gurgaon)": 2.1,
                "Cyber City (Gurgaon)": 1.6,
                "Sector 62 (Noida)": 0.9,
                "Greater Noida West": 0.6,
                "Vasant Kunj": 2.2,
                "Dwarka": 1.1,
                "Saket": 1.7,
                "Sector 150 (Noida)": 0.85
            }
        },
        "Hyderabad": {
            "state": "Telangana",
            "base_psf": 8200,
            "growth_rate_mean": 0.112,
            "localities": {
                "Jubilee Hills": 2.2,
                "Banjara Hills": 2.0,
                "Gachibowli": 1.3,
                "Hitec City": 1.35,
                "Kondapur": 1.05,
                "Madhapur": 1.25,
                "Kukatpally": 0.8,
                "Manikonda": 0.85
            }
        },
        "Pune": {
            "state": "Maharashtra",
            "base_psf": 7600,
            "growth_rate_mean": 0.092,
            "localities": {
                "Koregaon Park": 1.8,
                "Kalyani Nagar": 1.6,
                "Baner": 1.15,
                "Hinjewadi": 0.85,
                "Wakad": 0.9,
                "Kothrud": 1.25,
                "Viman Nagar": 1.3,
                "Hadapsar": 0.8
            }
        },
        "Chennai": {
            "state": "Tamil Nadu",
            "base_psf": 7400,
            "growth_rate_mean": 0.084,
            "localities": {
                "Adyar": 1.85,
                "Anna Nagar": 1.7,
                "Velachery": 1.05,
                "OMR (IT Corridor)": 0.85,
                "Thiruvanmiyur": 1.5,
                "Porur": 0.8,
                "Perungudi": 1.0,
                "Nungambakkam": 1.95
            }
        },
        "Kolkata": {
            "state": "West Bengal",
            "base_psf": 5800,
            "growth_rate_mean": 0.075,
            "localities": {
                "Alipore": 2.1,
                "Ballygunge": 1.9,
                "Salt Lake": 1.25,
                "New Town": 0.95,
                "Rajarhat": 0.8,
                "Jadavpur": 1.0,
                "Behala": 0.7,
                "EM Bypass": 1.15
            }
        },
        "Ahmedabad": {
            "state": "Gujarat",
            "base_psf": 5200,
            "growth_rate_mean": 0.089,
            "localities": {
                "Bodakdev": 1.6,
                "Satellite": 1.4,
                "SG Highway": 1.15,
                "Vastrapur": 1.3,
                "Prahlad Nagar": 1.35,
                "Bopal": 0.85,
                "Gota": 0.75,
                "Thaltej": 1.45
            }
        }
    }

    property_types = ["Apartment", "Villa", "Independent House", "Penthouse", "Builder Floor"]
    prop_type_weights = [0.65, 0.12, 0.10, 0.05, 0.08]
    prop_type_multipliers = {
        "Apartment": 1.0,
        "Villa": 1.45,
        "Independent House": 1.25,
        "Penthouse": 1.60,
        "Builder Floor": 1.10
    }

    facing_options = ["North", "East", "North-East", "West", "South", "South-East", "North-West", "South-West"]
    facing_weights = [0.25, 0.25, 0.20, 0.10, 0.08, 0.05, 0.04, 0.03]
    
    furnished_options = ["Unfurnished", "Semi-Furnished", "Fully Furnished"]
    furnished_weights = [0.25, 0.50, 0.25]
    
    owner_options = ["Individual", "Builder", "Agent"]
    owner_weights = [0.45, 0.35, 0.20]

    availability_options = ["Ready to Move", "Under Construction", "Resale"]
    availability_weights = [0.55, 0.30, 0.15]

    data = []

    cities = list(city_state_map.keys())
    city_weights = [0.22, 0.20, 0.18, 0.14, 0.10, 0.08, 0.04, 0.04]

    for i in range(1, n_samples + 1):
        prop_id = f"PROP_{i:06d}"
        
        # Select city & locality
        city = random.choices(cities, weights=city_weights, k=1)[0]
        city_info = city_state_map[city]
        state = city_info["state"]
        
        localities = list(city_info["localities"].keys())
        locality = random.choice(localities)
        loc_multiplier = city_info["localities"][locality]
        
        # Property type & BHK
        prop_type = random.choices(property_types, weights=prop_type_weights, k=1)[0]
        if prop_type == "Villa":
            bhk = random.choices([3, 4, 5], weights=[0.25, 0.50, 0.25], k=1)[0]
        elif prop_type == "Penthouse":
            bhk = random.choices([3, 4, 5], weights=[0.20, 0.50, 0.30], k=1)[0]
        elif prop_type == "Builder Floor":
            bhk = random.choices([2, 3, 4], weights=[0.30, 0.55, 0.15], k=1)[0]
        else:
            bhk = random.choices([1, 2, 3, 4, 5], weights=[0.12, 0.42, 0.34, 0.10, 0.02], k=1)[0]
        
        # Size calculation
        base_size_map = {1: 580, 2: 1050, 3: 1600, 4: 2500, 5: 3800}
        base_size = base_size_map[bhk]
        if prop_type in ["Villa", "Penthouse"]:
            base_size = int(base_size * 1.35)
        
        size_sqft = int(np.clip(np.random.normal(base_size, base_size * 0.12), bhk * 400, bhk * 1200))
        
        # Year built & Age
        year_built = random.choices(
            list(range(2000, 2026)),
            weights=[1 + (y - 2000) * 0.4 for y in range(2000, 2026)],
            k=1
        )[0]
        age_of_property = current_year - year_built
        
        # Floors
        if prop_type in ["Villa", "Independent House"]:
            total_floors = random.choice([2, 3, 4])
            floor_no = random.randint(0, total_floors)
        elif prop_type == "Builder Floor":
            total_floors = random.choice([3, 4, 5])
            floor_no = random.randint(1, total_floors)
        else:
            total_floors = random.choice([7, 12, 18, 24, 30, 40])
            floor_no = random.randint(0, total_floors)
            
        # Infrastructure features
        nearby_schools = random.randint(1, 15)
        nearby_hospitals = random.randint(1, 10)
        public_transport = random.choices(["High", "Medium", "Low"], weights=[0.45, 0.38, 0.17], k=1)[0]
        
        # Amenities
        amenity_tier = random.choices(["Basic", "Standard", "Premium", "Luxury"], weights=[0.25, 0.40, 0.25, 0.10], k=1)[0]
        amenities_map = {
            "Basic": "Security, Park",
            "Standard": "Gym, Park, Security, Power Backup",
            "Premium": "Gym, Swimming Pool, Clubhouse, CCTV, Kids Play Area",
            "Luxury": "Gym, Infinity Pool, Clubhouse, Tennis Court, Spa, Concierge, CCTV"
        }
        amenities = amenities_map[amenity_tier]
        amenity_multiplier = {"Basic": 0.95, "Standard": 1.0, "Premium": 1.08, "Luxury": 1.18}[amenity_tier]
        
        # Security
        security = random.choices(
            ["24x7 Security & CCTV", "Gated Security", "CCTV Only", "Guard Only"],
            weights=[0.55, 0.25, 0.12, 0.08],
            k=1
        )[0]
        
        parking_space = random.choices([0, 1, 2, 3], weights=[0.10, 0.55, 0.28, 0.07], k=1)[0]
        if bhk >= 3 and parking_space == 0:
            parking_space = random.choice([1, 2])
            
        facing = random.choices(facing_options, weights=facing_weights, k=1)[0]
        facing_mult = 1.03 if facing in ["East", "North", "North-East"] else 0.98
        
        furnished_status = random.choices(furnished_options, weights=furnished_weights, k=1)[0]
        furnished_mult = {"Unfurnished": 0.96, "Semi-Furnished": 1.0, "Fully Furnished": 1.08}[furnished_status]
        
        owner_type = random.choices(owner_options, weights=owner_weights, k=1)[0]
        availability_status = random.choices(availability_options, weights=availability_weights, k=1)[0]
        
        # Price Calculation Logic
        city_base = city_info["base_psf"]
        type_mult = prop_type_multipliers[prop_type]
        transport_mult = {"High": 1.06, "Medium": 1.0, "Low": 0.93}[public_transport]
        parking_mult = 1.0 + (parking_space * 0.02)
        age_depreciation = max(0.72, 1.0 - (age_of_property * 0.012))
        
        # Noise
        market_fluctuation = np.random.normal(1.0, 0.05)
        
        price_per_sqft = (city_base * loc_multiplier * type_mult * amenity_multiplier * 
                          facing_mult * furnished_mult * transport_mult * parking_mult * 
                          age_depreciation * market_fluctuation)
        
        # Add random slight outlier logic (1.5% of samples)
        if random.random() < 0.015:
            price_per_sqft *= random.choice([0.65, 1.45])
            
        price_per_sqft = round(price_per_sqft, 2)
        price_in_lakhs = round((price_per_sqft * size_sqft) / 100000.0, 2)
        
        # Calculate 5-year future price based on domain rules
        base_annual_growth = city_info["growth_rate_mean"]
        # Growth boosts:
        loc_boost = 0.015 if loc_multiplier > 1.2 else (-0.008 if loc_multiplier < 0.85 else 0.005)
        infra_boost = 0.008 if (nearby_schools >= 8 and nearby_hospitals >= 5) else 0.0
        transport_boost = 0.012 if public_transport == "High" else (-0.01 if public_transport == "Low" else 0.0)
        age_boost = 0.01 if age_of_property <= 3 else (-0.008 if age_of_property > 15 else 0.0)
        status_boost = 0.015 if availability_status == "Under Construction" else 0.0
        
        total_growth_rate = base_annual_growth + loc_boost + infra_boost + transport_boost + age_boost + status_boost
        total_growth_rate += np.random.normal(0, 0.008)  # slight noise
        total_growth_rate = float(np.clip(total_growth_rate, 0.04, 0.17))
        
        future_price_5y = round(price_in_lakhs * ((1.0 + total_growth_rate) ** 5), 2)
        
        # Good Investment Binary Target (Domain Rules as per PDF)
        # Factor 1: 5-Year CAGR >= 9.8% (demanding threshold)
        # Factor 2: Locality / PSF competitive value (relative to median)
        # Factor 3: Infrastructure & Transport >= Medium
        # Factor 4: Parking >= 1 & RERA / Developer tier
        cagr = (future_price_5y / price_in_lakhs) ** (1/5) - 1
        is_exceptional_cagr = cagr >= 0.105
        is_good_cagr = cagr >= 0.092
        is_high_infra = (nearby_schools >= 7 and nearby_hospitals >= 5)
        is_prime_transport = public_transport == "High"
        is_young_property = age_of_property <= 10
        
        score = 0
        if is_exceptional_cagr: score += 3
        elif is_good_cagr: score += 2
        
        if is_high_infra: score += 2
        if is_prime_transport: score += 2
        if parking_space and parking_space >= 1: score += 1
        if is_young_property: score += 1
        if bhk >= 2 and bhk <= 4: score += 1
        if availability_status == "Ready to Move": score += 1
        
        # Good investment threshold: score >= 7 gives balanced ~50/50 distribution
        good_investment = 1 if score >= 7 else 0
        
        # Add random 2% label noise for realistic real-estate ambiguity
        if random.random() < 0.03:
            good_investment = 1 - good_investment
            
        data.append({
            "ID": prop_id,
            "State": state,
            "City": city,
            "Locality": locality,
            "Property_Type": prop_type,
            "BHK": bhk,
            "Size_in_SqFt": size_sqft,
            "Price_in_Lakhs": price_in_lakhs,
            "Price_per_SqFt": price_per_sqft,
            "Year_Built": year_built,
            "Furnished_Status": furnished_status,
            "Floor_No": floor_no,
            "Total_Floors": total_floors,
            "Age_of_Property": age_of_property,
            "Nearby_Schools": nearby_schools,
            "Nearby_Hospitals": nearby_hospitals,
            "Public_Transport_Accessibility": public_transport,
            "Parking_Space": parking_space,
            "Security": security,
            "Amenities": amenities,
            "Facing": facing,
            "Owner_Type": owner_type,
            "Availability_Status": availability_status,
            "Future_Price_5Y": future_price_5y,
            "Good_Investment": good_investment
        })
        
    df = pd.DataFrame(data)
    
    # Introduce small realistic missingness (<0.8%) in optional columns to demonstrate robust data preprocessing
    missing_indices = np.random.choice(df.index, size=int(0.008 * len(df)), replace=False)
    df.loc[missing_indices[:len(missing_indices)//2], "Furnished_Status"] = np.nan
    df.loc[missing_indices[len(missing_indices)//2:], "Parking_Space"] = np.nan
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created at '{output_path}' with {len(df)} records and {len(df.columns)} columns.")
    print("Class distribution for 'Good_Investment':")
    print(df['Good_Investment'].value_counts(normalize=True))
    return df

if __name__ == "__main__":
    generate_housing_dataset()
