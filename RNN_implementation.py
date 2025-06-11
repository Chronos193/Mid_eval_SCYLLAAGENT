

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn

#device config
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

word='Add temperature sampling for more creative output'

#Get unique chars in word
vocab=sorted(set(word))

#Create mapping dicts
char_to_idx={char:idx for idx,char in enumerate(vocab)}
idx_to_char={idx:char for idx,char in enumerate(vocab)}
def word_to_tokens(word):
  return [char_to_idx[char] for char in word]
def tokens_to_word(tokens):
  return [idx_to_char[token] for token in tokens]

tokens=word_to_tokens(word)

input_seq=tokens[:-1]
target_seq=tokens[1:]

input_tensor = torch.tensor(input_seq).unsqueeze(0)   # shape (1, seq_len)
target_tensor = torch.tensor(target_seq).unsqueeze(0) # shape (1, seq_len)

input_tensor=input_tensor.to(device)
target_tensor=target_tensor.to(device)

# Hyperparmeters
embedding_dim=3
input_size=len(word)-1
hidden_size_1=8
hidden_size_2=4
num_epoch=2000
vocab_size=len(word)

class RNN(nn.Module):
  def __init__(self, input_size, embedding_dim, hidden_size_1,hidden_size_2):
    super(RNN, self).__init__()
    self.fc1=nn.Embedding(input_size, embedding_dim)
    self.fc2=nn.RNN(embedding_dim, hidden_size_1, batch_first=True)
    self.gelu=nn.GELU()
    self.fc3=nn.RNN(hidden_size_1, hidden_size_2,batch_first=True)
    self.output_layer = nn.Linear(hidden_size_2, vocab_size)
  def forward(self, x):
    out=self.fc1(x)
    out,hidden_state_1=self.fc2(out)
    out=self.gelu(out)
    out,hidden_state_2=self.fc3(out)
    out=self.output_layer(out)
    return out,hidden_state_1,hidden_state_2

model=RNN(input_size, embedding_dim, hidden_size_1,hidden_size_2)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

model=model.to(device)

# Training loop
for epoch in range(num_epoch):
  #Forward Prop
  output,hidden_state_1,hidden_state_2=model(input_tensor)
  loss = criterion(output.view(-1, vocab_size), target_tensor.view(-1))

  #Backward prop
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if (epoch+1)%100==0:
      print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f}')

output,_,_=model(input_tensor)

predicted_tokens=torch.argmax(output,dim=2)

predicted_tokens=predicted_tokens.squeeze(0)

predicted_word = ''.join([idx_to_char[token.item()] for token in predicted_tokens])

predicted_word

def generate_text(model, seed, word_to_tokens, tokens_to_word,generate_length):
  model.eval()

  input_seq = word_to_tokens(seed)
  input_tensor=torch.tensor(input_seq).unsqueeze(0).to(device)

  result = seed
  for i in range(generate_length):
    with torch.no_grad():
      output,_,_=model(input_tensor)

    #Get last time steps prediction
    last_logits=output[0,-1]
    next_token=torch.argmax(last_logits).item()
    next_char=idx_to_char[next_token]

    result+=next_char
    # Append the predicted token to input_tensor for next step
    input_tensor = torch.cat(
            [input_tensor, torch.tensor([[next_token]]).to(device)], dim=1
        )
  return result

seed='sampling '
generate_length=20
generate_text(model, seed, word_to_tokens, tokens_to_word,generate_length)

