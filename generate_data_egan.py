import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Load dataset
CSV_FILE = 'datos_reales_con_lado_y_ejercicio.csv'

df = pd.read_csv(CSV_FILE)

# Identify conditional columns and data columns
conditional_cols = ['tt_01', 'Ejercicio', 'Lado']
data_cols = [c for c in df.columns if c != 'H']

# Preprocess: one-hot encode categorical vars, scale numeric vars
categorical_cols = ['Ejercicio', 'Lado']
numeric_cols = [c for c in data_cols if c not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(feature_range=(-1, 1)), numeric_cols),
        ('cat', OneHotEncoder(sparse=False), categorical_cols),
    ]
)

X = preprocessor.fit_transform(df[data_cols])

# Determine sizes after preprocessing
input_dim = X.shape[1]
noise_dim = 32

# Define Generator
class Generator(nn.Module):
    def __init__(self, noise_dim, cond_dim, out_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.Tanh(),
        )

    def forward(self, noise, cond):
        inp = torch.cat([noise, cond], dim=1)
        return self.model(inp)

# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_dim, cond_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim + cond_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, cond):
        inp = torch.cat([x, cond], dim=1)
        return self.model(inp)

# Helper to extract conditional vector for each sample
cond_enc = preprocessor.named_transformers_['cat']
num_features = len(numeric_cols)
cond_dim = cond_enc.transform(df[categorical_cols]).shape[1] + 1  # +1 for tt_01

# Prepare tensors
scaled_data = torch.tensor(X, dtype=torch.float32)
conditions = torch.tensor(
    np.hstack([
        preprocessor.transform(df[conditional_cols])[:, num_features:],
        preprocessor.transform(df[['tt_01']])[:, 0:1],
    ]),
    dtype=torch.float32,
)

# Initialize models
generator = Generator(noise_dim, cond_dim, input_dim)
discriminator = Discriminator(input_dim, cond_dim)

criterion = nn.BCELoss()
optim_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
optim_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

def train_gan(epochs=1000, batch_size=32):
    for epoch in range(epochs):
        idx = np.random.permutation(len(scaled_data))
        for i in range(0, len(scaled_data), batch_size):
            batch_idx = idx[i:i+batch_size]
            real_data = scaled_data[batch_idx]
            cond_data = conditions[batch_idx]

            # Train Discriminator
            noise = torch.randn(len(batch_idx), noise_dim)
            fake_data = generator(noise, cond_data)

            real_labels = torch.ones(len(batch_idx), 1)
            fake_labels = torch.zeros(len(batch_idx), 1)

            optim_D.zero_grad()
            out_real = discriminator(real_data, cond_data)
            out_fake = discriminator(fake_data.detach(), cond_data)
            loss_D = criterion(out_real, real_labels) + criterion(out_fake, fake_labels)
            loss_D.backward()
            optim_D.step()

            # Train Generator
            optim_G.zero_grad()
            out_fake = discriminator(fake_data, cond_data)
            loss_G = criterion(out_fake, real_labels)
            loss_G.backward()
            optim_G.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs} - Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}')

def generate_samples(num_samples=9, cond_values=None):
    if cond_values is None:
        # Use random conditions from dataset
        idx = np.random.choice(len(df), num_samples)
        cond_vals = conditions[idx]
    else:
        # cond_values should be DataFrame with columns tt_01, Ejercicio, Lado
        tmp = preprocessor.transform(cond_values[conditional_cols])[:, num_features:]
        cond_vals = torch.tensor(
            np.hstack([tmp, preprocessor.transform(cond_values[['tt_01']])[:, 0:1]]),
            dtype=torch.float32,
        )
    noise = torch.randn(num_samples, noise_dim)
    with torch.no_grad():
        fake_scaled = generator(noise, cond_vals)
    fake_array = fake_scaled.numpy()
    # Inverse transform to original scale
    fake_data = preprocessor.inverse_transform(fake_array)
    # Build dataframe
    fake_df = pd.DataFrame(fake_data, columns=data_cols)
    return fake_df

if __name__ == '__main__':
    train_gan(epochs=1000)
    new_patients = generate_samples(9)
    print(new_patients.head())
    new_patients.to_csv('synthetic_patients.csv', index=False)
