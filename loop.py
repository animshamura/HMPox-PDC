from torch.utils.data import DataLoader

# Initialize dataset and dataloader
dataset = MpoxDataset('/path/to/real_mpox_dataset')
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Inference
with torch.no_grad():
    for genome, expr, prot_x, adj, input_ids, attention_mask, labels in loader:
        genome = genome.to(device)
        expr = expr.to(device)
        prot_x = prot_x.to(device)
        adj = adj.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        g_feat = genome_net(genome)
        e_feat = expr_net(expr)
        p_feat = prot_net(prot_x.squeeze(0), adj.squeeze(0)).mean(dim=0, keepdim=True)
        c_feat = clin_net(input_ids, attention_mask)

        detect, stage, treat = fusion_net(g_feat, e_feat, p_feat, c_feat)

        detect_score = detect.item()
        is_infected = detect_score > 0.5
        print(f"Mpox Detected: {'Yes' if is_infected else 'No'} (Score: {detect_score:.4f})")

        stage_probs = torch.softmax(stage, dim=1)
        predicted_stage = torch.argmax(stage_probs, dim=1).item()
        print(f"Predicted Stage: {predicted_stage} (Probabilities: {stage_probs.squeeze().tolist()})")

        treat_probs = torch.softmax(treat, dim=1)
        recommended_treat = torch.argmax(treat_probs, dim=1).item()
        print(f"Recommended Treatment: {treat_labels[recommended_treat]}")
        print(f"Details: {treat_descriptions[recommended_treat]}")
        print(f"Confidence Scores: {treat_probs.squeeze().tolist()}")
        break  # Remove for full batch processing
