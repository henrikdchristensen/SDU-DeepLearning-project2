import torch

def predict_on_fly(model, tokenizer, vocab, device, label_map, max_length):
    model.to(device)
    model.eval()

    print("Type 'q' to quit.")

    while True:
        # Get user input
        text = input("Enter a sentence: ").strip().lower()
        if text == "q":
            print("Exiting...")
            break

        # Tokenize the input text
        tokens = tokenizer.tokenize(text)

        # Convert tokens to IDs using the vocabulary
        token_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

        # Pad or truncate to match max_length
        if len(token_ids) < max_length:
            token_ids = token_ids + [vocab["<PAD>"]] * (max_length - len(token_ids))
        else:
            token_ids = token_ids[:max_length]
        
        # Convert to PyTorch tensor and move to device
        input_tensor = torch.tensor([token_ids]).to(device)

        # Predict using the model
        with torch.no_grad():
            outputs = model(input_tensor)  # Directly get raw logits or probabilities
            probabilities = torch.softmax(outputs, dim=1)  # Apply softmax to get probabilities
            predicted_label_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_label_idx].item()

        # Get the label name
        predicted_label = label_map[predicted_label_idx]

        # Print the result
        print(f"Predicted Label: {predicted_label} (Confidence: {confidence * 100:.2f}%)")
