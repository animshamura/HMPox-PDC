import torch
from models import CNNGenome, BiLSTMExpression, GCNProtein, BERTClinical, MpoxNetFusion
from transformers import BertTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize models
genome_net = CNNGenome(sequence_length=5000).to(device)
expr_net = BiLSTMExpression(input_size=50, hidden_size=128).to(device)
prot_net = GCNProtein(input_dim=64, hidden_dim=128, output_dim=128).to(device)
clin_net = BERTClinical().to(device)
fusion_net = MpoxNetFusion(dims=[128, 128, 128, 768]).to(device)

# Example dummy input for BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
clinical_text = "Patient has fever, rash and lymphadenopathy"
clinical_tokens = tokenizer(clinical_text, return_tensors='pt', padding=True, truncation=True).to(device)

# Genome input (dummy data)
genome_input = torch.randn(1, 5000).view(1, 1, 5000).float().to(device)  # [batch, channels, sequence]

# Expression input (dummy data)
expr_input = torch.randn(1, 100, 50).to(device)  # [batch, time_steps, features]

# Protein network input (dummy data)
num_nodes = 10
prot_x = torch.randn(num_nodes, 64).to(device)  # [nodes, features]
prot_edge_index = torch.randint(0, num_nodes, (2, 50)).to(device)  # random graph

# Create adjacency matrix for the graph
adj = torch.zeros(num_nodes, num_nodes).to(device)
row, col = prot_edge_index
adj[row, col] = 1
adj[col, row] = 1  # undirected graph

# Forward pass through individual models
with torch.no_grad():
    g_feat = genome_net(genome_input)  # [1, 128]
    e_feat = expr_net(expr_input)      # [1, 128]
    p_feat = prot_net(prot_x, adj)     # [num_nodes, 128]
    c_feat = clin_net(clinical_tokens["input_ids"], clinical_tokens["attention_mask"])  # [1, 768]

    # Average protein node features into a single vector
    p_feat = p_feat.mean(dim=0, keepdim=True)  # [1, 128]

    # Ensure all shapes match
    print("g_feat:", g_feat.shape)    # [1, 128]
    print("e_feat:", e_feat.shape)    # [1, 128]
    print("p_feat:", p_feat.shape)    # [1, 128]
    print("c_feat:", c_feat.shape)    # [1, 768]

    # Concatenate features: [1, 1152]
   # Check individual feature shapes
print("g_feat:", g_feat.shape)    # [1, 128]
print("e_feat:", e_feat.shape)    # [1, 128]
print("p_feat:", p_feat.shape)    # [1, 128]
print("c_feat:", c_feat.shape)    # [1, 768]

# Pass through fusion net using separate inputs
detect, stage, treat = fusion_net(g_feat, e_feat, p_feat, c_feat)


# Output results
print("Detection:", detect)
print("Stage Prediction:", stage)
print("Treatment Prediction:", treat)

detect_score = detect.item()
is_infected = detect_score > 0.5
print(f"Mpox Detected: {'Yes' if is_infected else 'No'} (Score: {detect_score:.4f})")

# Stage prediction
stage_probs = torch.softmax(stage, dim=1)
predicted_stage = torch.argmax(stage_probs, dim=1).item()
print(f"Predicted Stage: {predicted_stage} (Probabilities: {stage_probs.squeeze().tolist()})")

# Treatment prediction
treat_labels = {
    0: "Supportive Care Only",
    1: "Tecovirimat Protocol",
    2: "Intensive Antiviral & Immuno-Supportive Regimen"
}

treat_descriptions = {
    0: "Hydration, antipyretics (for fever), antihistamines for itching. Used for mild or early-stage Mpox. No antivirals needed unless complications arise.",
    1: "Includes oral Tecovirimat (TPOXX), antiviral therapy especially for moderate Mpox or patients with risk factors (HIV+, immunocompromised, etc.). Monitor liver and kidney function during course.",
    2: "Severe or late-stage Mpox cases with complications. Combines Tecovirimat with Brincidofovir, IV hydration, immune-boosting therapy (e.g., interferons), possible hospitalization."
}

treat_probs = torch.softmax(treat, dim=1)
recommended_treat = torch.argmax(treat_probs, dim=1).item()

print(f"Recommended Treatment Plan: {treat_labels[recommended_treat]}")
print(f"Details: {treat_descriptions[recommended_treat]}")
print(f"Confidence Scores: {treat_probs.squeeze().tolist()}")
