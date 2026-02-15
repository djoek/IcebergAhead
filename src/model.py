import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load and preprocess Titanic dataset
def load_titanic_data():
    import seaborn as sns
    df = sns.load_dataset('titanic')

    # Create a clean copy to avoid chained assignment
    df = df.copy()

    # Map sex to numeric (correct column name is 'sex', not 'Sex')
    df['sex_numeric'] = df['sex'].map({'male': 0, 'female': 1})

    # Fill missing values
    df['age'] = df['age'].fillna(df['age'].median())
    df['fare'] = df['fare'].fillna(df['fare'].median())
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

    # Create family size features
    df['family_size'] = df['sibsp'] + df['parch']
    df['is_alone'] = (df['family_size'] == 0).astype(int)

    # Embarked one-hot encoding
    df['embarked_Q'] = (df['embarked'] == 'Q').astype(int)
    df['embarked_S'] = (df['embarked'] == 'S').astype(int)

    # Select 8 features - all column names are lowercase in seaborn's dataset
    feature_columns = [
        'pclass',  # Passenger class (1, 2, 3)
        'sex_numeric',  # Sex (0=male, 1=female)
        'age',  # Age in years
        'fare',  # Ticket fare
        'family_size',  # Number of family members aboard
        'is_alone',  # Whether passenger is alone
        'embarked_Q',  # Embarked at Queenstown
        'embarked_S'  # Embarked at Southampton
    ]

    # Create features DataFrame
    features = df[feature_columns].copy()
    target = df['survived'].copy()

    # Remove any rows with NaN (should be none after our filling, but just in case)
    mask = ~features.isna().any(axis=1) & ~target.isna()
    features = features[mask]
    target = target[mask]

    print(f"Dataset loaded: {len(features)} passengers")
    print(f"Features: {list(features.columns)}")
    print(f"Survival rate: {target.mean():.2%}")

    return features.values, target.values


class BoundedTitanicNet(nn.Module):

    def __init__(self, weight_min=-2.0, weight_max=2.0):
        super(BoundedTitanicNet, self).__init__()
        self.inputs = nn.Linear(8, 4)
        self.layer1 = nn.Linear(4, 4)
        self.layer2 = nn.Linear(4, 4)
        self.layer3 = nn.Linear(4, 4)
        self.layer4 = nn.Linear(4, 4)
        self.output = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

        # Store bounds
        self.weight_min = weight_min
        self.weight_max = weight_max

        # Initialize weights within bounds
        self._init_bounded_weights()

    def _init_bounded_weights(self):
        """Initialize all weights uniformly in [weight_min, weight_max]"""
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            nn.init.uniform_(layer.weight, self.weight_min, self.weight_max)
            nn.init.uniform_(layer.bias, self.weight_min, self.weight_max)

    def clamp_weights(self):
        """Hard clamp all weights to bounds"""
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.output]:
            layer.weight.data.clamp_(self.weight_min, self.weight_max)
            layer.bias.data.clamp_(self.weight_min, self.weight_max)

    def forward(self, x):
        x = torch.relu(self.inputs(x))
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = self.sigmoid(self.output(x))
        return x


# Training function (same as before)
def train_model(model, X_train, y_train, X_test, y_test, epochs=100):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).reshape(-1, 1)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        model.clamp_weights()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_t)
                test_loss = criterion(test_outputs, y_test_t)
                predictions = (test_outputs > 0.5).float()
                accuracy = (predictions == y_test_t).float().mean()

                print(f'Epoch [{epoch + 1}/{epochs}], '
                      f'Train Loss: {loss.item():.4f}, '
                      f'Test Loss: {test_loss.item():.4f}, '
                      f'Accuracy: {accuracy.item():.4f}')

    return model

# Test cases: [pclass, sex, age, fare, family_size, is_alone, embarked_Q, embarked_S]
test_cases = [
    {
        'desc': 'Rich young woman in 1st class',
        'features': [1, 1, 25, 100, 0, 1, 0, 0]
    },
    {
        'desc': 'Poor old man in 3rd class',
        'features': [3, 0, 60, 7, 2, 0, 0, 1]
    },
    {
        'desc': 'Middle-aged man in 2nd class',
        'features': [2, 0, 35, 25, 1, 0, 0, 1]
    },
    {
        'desc': 'Young girl in 3rd class with family',
        'features': [3, 1, 8, 15, 3, 0, 0, 1]
    }
]

# Main execution
if __name__ == "__main__":
    print("Loading Titanic dataset...")
    X, y = load_titanic_data()

    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples\n")

    print("Creating and training model...")
    model = BoundedTitanicNet()
    model = train_model(model, X_train, y_train, X_test, y_test, epochs=600)

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)

    # Example predictions
    print("\nExample predictions:")

    model.eval()
    with torch.no_grad():
        for case in test_cases:
            test_input = scaler.transform([case['features']])
            test_tensor = torch.FloatTensor(test_input)
            survival_prob = model(test_tensor).item()
            print(f"{case['desc']:40s} → {survival_prob:.2%} survival")

    # After training
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
    }, './titanic_model.pth')

