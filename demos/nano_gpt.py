import torch
import torch.nn as nn
import torch.nn.functional as F

# Sample text corpus
text = "hello world this is a small GPT demo GPT model generates text"

# Simple tokenizer
words = sorted(set(text.split()))
word_to_token = {word: idx for idx, word in enumerate(words)}
token_to_word = {idx: word for word, idx in word_to_token.items()}
vocab_size = len(words)
print("Vocabulary size:", vocab_size)

print("-------------------")

# Tokenize text
encoded = [word_to_token[word] for word in text.split()]
data = torch.tensor(encoded)

# Model Hyperparameters
embed_size = 16
num_heads = 2
block_size = 4
num_layers = 1


# Minimal GPT model
class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(block_size, embed_size)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embed(idx)  # get token embeddings by index
        pos_emb = self.pos_embed(torch.arange(T))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        logits = self.fc_out(x)
        return logits


# Instantiate model
model = MiniGPT()

# Sample Input (batch_size=1, block_size=4)
context = data[:block_size].unsqueeze(0)
print("Input context:", context)  # (batch_size, block_size)
print("Input words:", [token_to_word[idx.item()] for idx in context[0]])

# Forward Pass
logits = model(context)
print("Output logits shape:", logits.shape)  # (batch_size, block_size, vocab_size)

# Predict next token (greedy)
next_token_logits = logits[0, -1]
predicted_token = torch.argmax(F.softmax(next_token_logits, dim=0)).item()
print("Predicted next word:", token_to_word[predicted_token])

# Decode tokens back to words (for demonstration)
output_tokens = torch.argmax(logits, dim=-1).squeeze().tolist()
output_words = [token_to_word[tok] for tok in output_tokens]
print("Output words from model:", output_words)


print("-------------------")


# Function to generate new text
def generate(model, context, max_new_tokens):
    model.eval()
    generated = context.tolist()[0]  # Starting tokens
    context = context.clone()

    for _ in range(max_new_tokens):
        # Limit context size to block_size
        context_cond = context[:, -block_size:]

        # Forward pass
        logits = model(context_cond)

        # Get the logits for the last token and apply softmax
        next_token_logits = logits[:, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)

        # Greedy sampling (argmax)
        next_token = torch.argmax(probs, dim=-1).unsqueeze(0)

        # Append predicted token to the generated sequence
        generated.append(next_token.item())

        # Append predicted token to context for next step
        context = torch.cat((context, next_token), dim=1)

    return generated


# Initial context
context = data[:block_size].unsqueeze(0)

# Generate 5 new tokens
generated_tokens = generate(model, context, max_new_tokens=5)

# Map tokens back to words
generated_words = [token_to_word[tok] for tok in generated_tokens]
print("Generated words:", generated_words)
