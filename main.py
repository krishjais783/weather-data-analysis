import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------- load data -------------------------------------
df = pd.read_csv("weather_data.csv")
print(df.shape)

# ------------------------------------- check missing -------------------------------------
print(df.isnull().sum())

# ------------------------------------- duplicates -------------------------------------
print(df.duplicated().sum())
df = df.drop_duplicates()
print(df.duplicated().sum())

# ------------------------------------- clean column names -------------------------------------
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace('(', '')
df.columns = df.columns.str.replace(')', '')

# ------------------------------------- Renaming -------------------------------------
df.columns = df.columns.str.replace('temp_meanc', 'temp_mean')
df.columns = df.columns.str.replace('temp_minc', 'temp_min')
df.columns = df.columns.str.replace('temp_maxc', 'temp_max')

print(df.columns)

# ------------------------------------- save cleaned file -------------------------------------
df.to_csv("clean_weather_data.csv", index=False)
print("done cleaning")

# ------------------------------------- analysis -------------------------------------
print(df.describe())

print("avg temp:", df['temp_mean'].mean())
print("max temp:", df['temp_max'].max())
print("min temp:", df['temp_min'].min())

# average wind speed
print("avg wind speed:", df['wind_speed'].mean())

# correlation
print(df.corr(numeric_only=True))

# temp correlation focus
print(df.corr(numeric_only=True)['temp_mean'].sort_values(ascending=False))

# ------------------------------------- graphs -------------------------------------
sns.set_style("whitegrid")

# ------------------------------------- temp distribution -------------------------------------
plt.figure()
sns.histplot(df['temp_mean'], bins=30)
plt.title("Temperature Distribution")
plt.show()

# ------------------------------------- temp vs radiation -------------------------------------
plt.figure()
sns.scatterplot(x=df['global_radiation'], y=df['temp_mean'])
plt.title("Temp vs Radiation")
plt.show()

# ------------------------------------- heatmap -------------------------------------
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Heatmap")
plt.tight_layout()
plt.show()

# ------------------------------------- wind distribution -------------------------------------
plt.figure()
sns.histplot(df['wind_speed'], bins=30)
plt.title("Wind Speed")
plt.show()

# ------------------------------------- Insights -------------------------------------
print("\n--- Key Insights ---")
print("Temperature is mostly concentrated in a moderate range.")
print("There is a strong positive correlation between solar radiation and temperature.")
print("Wind speed has minimal impact on temperature.")

