import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from lime.lime_text import LimeTextExplainer
from torch.amp import GradScaler, autocast
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import random

# Set matplotlib backend to 'Agg' for non-interactive environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
seed = 35
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

# Load datasets
train_df = pd.read_csv('train_dataV3.csv')
validation_df = pd.read_csv('valid_LIAR2.csv')
test_df = pd.read_csv('test_LIAR2.csv')

# Load pre-trained RoBERTa model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
num_labels = 6

# Reduce max_length parameters to decrease memory usage
MAX_LENGTH_STATEMENT = 128  # You can try reducing this further
MAX_LENGTH_SUBJECT = 16
MAX_LENGTH_CONTEXT = 16
MAX_LENGTH_JUSTIFICATION = 128  # You can try reducing this further

# Tokenize and prepare dataset
def prepare_dataset(df, tokenizer):
    df = df.fillna('')
    df = df.astype(str)
    # Tokenize each feature separately
    statement_encodings = tokenizer(
        df['statement'].tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH_STATEMENT,
        return_tensors="pt"
    )
    subject_encodings = tokenizer(
        df['subject'].tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH_SUBJECT,
        return_tensors="pt"
    )
    context_encodings = tokenizer(
        df['context'].tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH_CONTEXT,
        return_tensors="pt"
    )
    justification_encodings = tokenizer(
        df['justification'].tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH_JUSTIFICATION,
        return_tensors="pt"
    )

    # Concatenate the input IDs and attention masks
    input_ids = torch.cat([
        statement_encodings['input_ids'],
        subject_encodings['input_ids'],
        context_encodings['input_ids'],
        justification_encodings['input_ids']
    ], dim=1)

    attention_mask = torch.cat([
        statement_encodings['attention_mask'],
        subject_encodings['attention_mask'],
        context_encodings['attention_mask'],
        justification_encodings['attention_mask']
    ], dim=1)

    labels = pd.to_numeric(df['label'], errors='coerce').fillna(-1).astype(int)
    labels = torch.tensor(labels.values)
    return TensorDataset(input_ids, attention_mask, labels)

# Prepare datasets
train_dataset = prepare_dataset(train_df, tokenizer)
val_dataset = prepare_dataset(validation_df, tokenizer)
test_dataset = prepare_dataset(test_df, tokenizer)

# DataLoaders
batch_size = 16  # Reduced from 32 to decrease memory usage
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define model, optimizer, and scheduler
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=num_labels,
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.3
)
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.001)
num_epochs = 4
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)
loss_fn = torch.nn.CrossEntropyLoss()

# Enable mixed precision training
scaler = GradScaler()

# Training function with mixed precision
def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model(input_ids, attention_mask=attention_mask).logits
            loss = loss_fn(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()

        # Clear cache to free up memory
        del input_ids, attention_mask, labels, outputs, loss
        torch.cuda.empty_cache()

    return total_loss / len(data_loader)

# Evaluation function
def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(input_ids, attention_mask=attention_mask).logits
                loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            # Clear cache to free up memory
            del input_ids, attention_mask, labels, outputs, loss, predicted
            torch.cuda.empty_cache()
    return total_loss / len(data_loader), all_labels, all_preds

# Training loop with loss tracking and early stopping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_losses = []
val_losses = []

# Early stopping parameters
early_stopping_patience = 3
best_val_loss = float('inf')
early_stopping_counter = 0

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, device, scaler)
    val_loss, _, _ = evaluate(model, val_loader, loss_fn, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        # Save the best model and training state
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': val_loss
        }, 'best_model_checkpoint_V2.pth')
        print("Validation loss improved. Model and state saved.")
    else:
        early_stopping_counter += 1
        print(f"No improvement in validation loss for {early_stopping_counter} epoch(s).")

    if early_stopping_counter >= early_stopping_patience:
        print("Early stopping triggered. Stopping training.")
        break

# Load the best model and state for final evaluation
checkpoint = torch.load('best_model_checkpoint_V2.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
model.to(device)

# Plot training and validation loss curves
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve_roberta_extra_features.png')
plt.close()

# Evaluate on training, validation, and test sets
def display_results(model, data_loader, dataset_name):
    loss, all_labels, all_preds = evaluate(model, data_loader, loss_fn, device)
    accuracy = (np.array(all_labels) == np.array(all_preds)).mean() * 100
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Get classification report as a dictionary
    class_report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=[f"Class {i}" for i in range(num_labels)],
        output_dict=True,
        digits=2
    )

    # Get string version of classification report for display
    class_report = classification_report(
        all_labels,
        all_preds,
        target_names=[f"Class {i}" for i in range(num_labels)],
        digits=2
    )

    # Calculate per-label accuracy and F1 score
    per_label_accuracy = {}
    per_label_f1 = {}
    for i, class_name in enumerate([f"Class {i}" for i in range(num_labels)]):
        true_positives = conf_matrix[i, i]
        total_actual = conf_matrix[i].sum()
        per_label_accuracy[class_name] = (true_positives / total_actual * 100) if total_actual > 0 else 0.0

        # Per-label F1 score
        precision = class_report_dict[class_name]["precision"]
        recall = class_report_dict[class_name]["recall"]
        per_label_f1[class_name] = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Extract total F1 score (weighted average)
    total_f1_score = class_report_dict["weighted avg"]["f1-score"] * 100

    print(f"\n{dataset_name} Loss: {loss:.4f}")
    print(f"{dataset_name} Accuracy: {accuracy:.2f}%")
    print(f"{dataset_name} Multi-Class Confusion Matrix:")
    print(conf_matrix)
    print(f"{dataset_name} Classification Report:")
    print(class_report)

    # Print per-label accuracy
    print(f"{dataset_name} Accuracy Per Label:")
    for label, acc in per_label_accuracy.items():
        print(f"  {label}: {acc:.2f}%")

    # Print per-label F1 score
    print(f"{dataset_name} F1 Score Per Label:")
    for label, f1 in per_label_f1.items():
        print(f"  {label}: {f1:.3f}")

    # Print total F1 score
    print(f"{dataset_name} Total F1 Score: {total_f1_score:.2f}%")

# Display final results
print("\nFinal Evaluation Results:")
display_results(model, train_loader, "Training")
display_results(model, val_loader, "Validation")
display_results(model, test_loader, "Test")

from collections import defaultdict
from tqdm import tqdm

def analyze_feature_importance_per_class(model, tokenizer, test_df, device, num_instances_per_class=100):
    """
    Perform feature importance analysis per class using LIME.

    Args:
        model: Trained model for prediction.
        tokenizer: Tokenizer for text processing.
        test_df: Test dataset.
        device: Device for computations (CPU/GPU).
        num_instances_per_class: Number of instances per class to analyze.

    Returns:
        feature_importance_per_class: Dictionary of feature importances per class.
    """
    # Ensure the label column is numeric
    test_df['label'] = pd.to_numeric(test_df['label'], errors='coerce').astype(int)

    # Define class names as just the numeric labels (0, 1, 2, etc.)
    class_names = [str(i) for i in range(num_labels)]

    # Initialize LIME explainer
    explainer = LimeTextExplainer(class_names=class_names)
    feature_importance_per_class = defaultdict(lambda: defaultdict(float))

    # Move model to CPU for LIME processing
    model_cpu = model.cpu()
    device_cpu = torch.device('cpu')
    torch.cuda.empty_cache()

    print("Starting LIME analysis...")
    for class_label in range(num_labels):
        print(f"\nAnalyzing Class {class_label}...")

        # Select instances for the current class
        class_instances = test_df[test_df['label'] == class_label]
        num_samples = min(num_instances_per_class, len(class_instances))
        
        if num_samples == 0:
            print(f"No samples available for Class {class_label}. Skipping...")
            continue

        sampled_instances = class_instances.sample(num_samples, random_state=seed)

        for _, row in tqdm(sampled_instances.iterrows(), total=num_samples):
            statement = row['statement']
            subject = row['subject']
            context = row['context']
            justification = row['justification']

            # Define prediction function for LIME
            def predict_proba_statements(statements):
                df = pd.DataFrame({
                    'statement': statements,
                    'subject': [subject] * len(statements),
                    'context': [context] * len(statements),
                    'justification': [justification] * len(statements),
                })
                df = df.fillna('')
                df = df.astype(str)
                # Tokenize each feature separately
                statement_encodings = tokenizer(
                    df['statement'].tolist(),
                    truncation=True,
                    padding=True,
                    max_length=MAX_LENGTH_STATEMENT,
                    return_tensors="pt"
                )
                subject_encodings = tokenizer(
                    df['subject'].tolist(),
                    truncation=True,
                    padding=True,
                    max_length=MAX_LENGTH_SUBJECT,
                    return_tensors="pt"
                )
                context_encodings = tokenizer(
                    df['context'].tolist(),
                    truncation=True,
                    padding=True,
                    max_length=MAX_LENGTH_CONTEXT,
                    return_tensors="pt"
                )
                justification_encodings = tokenizer(
                    df['justification'].tolist(),
                    truncation=True,
                    padding=True,
                    max_length=MAX_LENGTH_JUSTIFICATION,
                    return_tensors="pt"
                )
                # Concatenate input IDs and attention masks
                input_ids = torch.cat([
                    statement_encodings['input_ids'],
                    subject_encodings['input_ids'],
                    context_encodings['input_ids'],
                    justification_encodings['input_ids']
                ], dim=1).to(device_cpu)
                attention_mask = torch.cat([
                    statement_encodings['attention_mask'],
                    subject_encodings['attention_mask'],
                    context_encodings['attention_mask'],
                    justification_encodings['attention_mask']
                ], dim=1).to(device_cpu)
                # Get model outputs
                with torch.no_grad():
                    outputs = model_cpu(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    probs = torch.nn.functional.softmax(logits, dim=1)
                del input_ids, attention_mask, outputs, logits
                return probs.cpu().numpy()

            # Generate LIME explanation
            exp = explainer.explain_instance(
                statement,
                predict_proba_statements,
                num_features=20,  # Focus on top 20 features
                num_samples=200,  # Number of perturbations for LIME
                labels=[class_label]
            )

            # Aggregate feature importance for the current class
            for feature, importance in exp.as_list(label=class_label):
                feature_importance_per_class[class_label][feature] += importance

    # Move model back to original device
    model.to(device)
    torch.cuda.empty_cache()

    # Sort features by importance within each class
    for class_label in feature_importance_per_class:
        feature_importance_per_class[class_label] = sorted(
            feature_importance_per_class[class_label].items(),
            key=lambda x: x[1],
            reverse=True
        )
    
    print("LIME analysis complete.")
    return feature_importance_per_class

# Perform LIME analysis
feature_importance = analyze_feature_importance_per_class(
    model, tokenizer, test_df, device, num_instances_per_class=100
)

# Print top words for each class
for class_label, features in feature_importance.items():
    print(f"\nTop words for Class {class_label}:")
    for word, score in features[:20]:  # Top 20 features
        print(f"  {word}: {score:.4f}")

analyze_feature_importance_per_class(model, tokenizer, test_df, device, num_instances_per_class=100)