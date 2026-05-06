# ============================================
# NETFLIX DATA CLEANING & VISUALIZATION PROJECT
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create visuals folder
os.makedirs("visuals", exist_ok=True)

# Style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# ============================================
# LOAD DATASET
# ============================================

df = pd.read_csv("Data Cleaning/netflix_titles.csv")

print("\n========== FIRST 5 ROWS ==========\n")
print(df.head())

print("\n========== DATASET INFO ==========\n")
print(df.info())

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

# ============================================
# DATA CLEANING
# ============================================

# Remove duplicates
df.drop_duplicates(inplace=True)

# Fill missing values
df["director"] = df["director"].fillna("Unknown")
df["cast"] = df["cast"].fillna("Not Available")
df["country"] = df["country"].fillna("Unknown")
df["rating"] = df["rating"].fillna(df["rating"].mode()[0])

# Drop rows where date_added is missing
df.dropna(subset=["date_added"], inplace=True)

# Convert date column
df["date_added"] = df["date_added"].str.strip()

df["date_added"] = pd.to_datetime(
    df["date_added"],
    format="mixed",
    errors="coerce"
)

# Extract year and month
df["year_added"] = df["date_added"].dt.year
df["month_added"] = df["date_added"].dt.month

# Convert duration into numeric values
df["duration_number"] = df["duration"].str.extract(r"(\d+)").astype(float)

print("\n========== CLEANED DATA ==========\n")
print(df.head())

# ============================================
# OUTLIER DETECTION
# ============================================

plt.figure(figsize=(10, 5))
sns.boxplot(x=df["duration_number"])

plt.title("Outlier Detection for Duration")
plt.xlabel("Duration")
plt.savefig("visuals/outlier_detection.png")
plt.show()

# ============================================
# VISUALIZATION 1
# CONTENT TYPE DISTRIBUTION
# ============================================

plt.figure(figsize=(8, 5))

sns.countplot(
    x="type",
    data=df,
    palette="Set2"
)

plt.title("Movies vs TV Shows on Netflix")
plt.xlabel("Content Type")
plt.ylabel("Count")

plt.savefig("visuals/content_type_distribution.png")
plt.show()

# ============================================
# VISUALIZATION 2
# TOP 10 COUNTRIES
# ============================================

top_countries = df["country"].value_counts().head(10)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=top_countries.values,
    y=top_countries.index,
    palette="viridis"
)

plt.title("Top 10 Content Producing Countries")
plt.xlabel("Number of Shows")
plt.ylabel("Country")

plt.savefig("visuals/top_countries.png")
plt.show()

# ============================================
# VISUALIZATION 3
# CONTENT GROWTH OVER YEARS
# ============================================

content_growth = df["year_added"].value_counts().sort_index()

plt.figure(figsize=(12, 6))

sns.lineplot(
    x=content_growth.index,
    y=content_growth.values,
    marker="o"
)

plt.title("Netflix Content Growth Over Years")
plt.xlabel("Year")
plt.ylabel("Number of Releases")

plt.savefig("visuals/content_growth.png")
plt.show()

# ============================================
# VISUALIZATION 4
# RATINGS DISTRIBUTION
# ============================================

plt.figure(figsize=(12, 6))

sns.countplot(
    y="rating",
    data=df,
    order=df["rating"].value_counts().index,
    palette="coolwarm"
)

plt.title("Content Ratings Distribution")
plt.xlabel("Count")
plt.ylabel("Rating")

plt.savefig("visuals/rating_distribution.png")
plt.show()

# ============================================
# VISUALIZATION 5
# GENRE ANALYSIS
# ============================================

genres = df["listed_in"].str.split(",", expand=True).stack()

top_genres = genres.value_counts().head(10)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=top_genres.values,
    y=top_genres.index,
    palette="magma"
)

plt.title("Top 10 Genres on Netflix")
plt.xlabel("Count")
plt.ylabel("Genre")

plt.savefig("visuals/genre_analysis.png")
plt.show()

# ============================================
# KEY INSIGHTS
# ============================================

print("\n========== KEY INSIGHTS ==========\n")

print(f"Total Content: {len(df)}")

print(f"Most Common Rating: {df['rating'].mode()[0]}")

print(f"Top Producing Country: {df['country'].value_counts().idxmax()}")

print(f"Most Popular Genre: {top_genres.index[0]}")

print("\nProject Completed Successfully!")