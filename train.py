from torch.optim import Adam
from vae import loss_function

def train_vae(model, dataloader, epochs=10, learning_rate=1e-3, beta=1.0):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    
    for epoch in range(epochs):
        train_loss = 0
        for batch_data in dataloader:
            # If your dataset provides data in the form (data, labels), use batch_data[0]
            # If not, just use batch_data
            data = batch_data[0].float()
            
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar, beta)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(dataloader.dataset)}")
