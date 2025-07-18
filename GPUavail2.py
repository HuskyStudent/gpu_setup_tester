import torch
import torch.nn as nn
import torch.optim as optim
import time

def train_on(device):
    print(f"\n🔧 Training on: {device}")
    
    # Dummy data
    X = torch.randn(10000, 100).to(device)
    y = torch.randint(0, 2, (10000,)).to(device)

    # Model
    model = nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    start = time.time()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0 or epoch == 100 -1:
            print(f"Epoch {epoch+1}/100, Loss: {loss.item():.4f}")
    end = time.time()
    
    print(f"⏱ Completed in {end - start:.4f} seconds on {device}")

print("hi")

# Run both
train_on(torch.device("cpu"))

if torch.cuda.is_available():
    train_on(torch.device("cuda"))
    print("GPU is available")
else:
    print("\n❌ CUDA not available.")
