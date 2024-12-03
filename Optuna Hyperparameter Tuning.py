import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import random
import json

# Set random seeds for reproducibility
seed = 35
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

# Load datasets
train_df_expanded = pd.read_csv('expanded_train_df.csv')
validation_df = pd.read_csv('valid_LIAR2.csv')
test_df = pd.read_csv('test_LIAR2.csv')

# Load pre-trained RoBERTa model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
num_labels = 6

# Tokenize and prepare dataset
def prepare_dataset(df, tokenizer):
    df = df.fillna('')
    df = df.astype(str)
    expanded_statement_encodings = tokenizer(df['expanded_statement'].tolist(), truncation=True, padding=True, max_length=192, return_tensors="pt")
    subject_encodings = tokenizer(df['subject'].tolist(), truncation=True, padding=True, max_length=32, return_tensors="pt")
    context_encodings = tokenizer(df['context'].tolist(), truncation=True, padding=True, max_length=32, return_tensors="pt") 
    justification_encodings = tokenizer(df['justification'].tolist(), truncation=True, padding=True, max_length=256, return_tensors="pt")
    
     # Concatenate the input IDs and attention masks
    input_ids = torch.cat([expanded_statement_encodings['input_ids'], 
                           subject_encodings['input_ids'],  
                           context_encodings['input_ids'],
                           justification_encodings['input_ids']], dim=1)
    
    attention_mask = torch.cat([expanded_statement_encodings['attention_mask'],  
                                subject_encodings['attention_mask'],  
                                context_encodings['attention_mask'],
                                justification_encodings['input_ids']], dim=1)

    labels = pd.to_numeric(df['label'], errors='coerce').fillna(-1).astype(int)
    labels = torch.tensor(labels.values)
    return TensorDataset(input_ids, attention_mask, labels)

validation_df = validation_df.rename(columns={'statement': 'expanded_statement'})
test_df = test_df.rename(columns={'statement': 'expanded_statement'})

validation_df = validation_df.rename(columns={'statement': 'expanded_statement'})
test_df = test_df.rename(columns={'statement': 'expanded_statement'})

# Prepare datasets
train_dataset = prepare_dataset(train_df_expanded, tokenizer)
val_dataset = prepare_dataset(validation_df, tokenizer)
test_dataset = prepare_dataset(test_df, tokenizer)

# DataLoader with dynamic batch size
def create_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Train function for a single epoch
def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask).logits
        loss = loss_fn(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
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
            outputs = model(input_ids, attention_mask=attention_mask).logits
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    return total_loss / len(data_loader), all_labels, all_preds

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameter tuning
    batch_size = trial.suggest_int('batch_size', 16, 96, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    num_epochs = trial.suggest_int('num_epochs', 5, 25)

    # Create DataLoader with suggested batch size
    train_loader = create_dataloader(train_dataset, batch_size)
    val_loader = create_dataloader(val_dataset, batch_size)
    
    # Define the model with suggested dropout rate
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=num_labels,
        hidden_dropout_prob=dropout_rate,
        attention_probs_dropout_prob=dropout_rate
    )
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training and evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train the model for the suggested number of epochs
    best_val_loss = float('inf')
    early_stopping_counter = 0
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, device)
        val_loss, _, _ = evaluate(model, val_loader, loss_fn, device)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        if early_stopping_counter >= 5:
            break

    return best_val_loss

# Set up the Optuna study
study = optuna.create_study(direction='minimize')  # We want to minimize validation loss
study.optimize(objective, n_trials=10)

# Save the best hyperparameters to a JSON file
best_hyperparameters = study.best_params
with open('best_hyperparameters_RoBERTa_justification.json', 'w') as f:
    json.dump(best_hyperparameters, f, indent=4)

# Output the best hyperparameters
print(f"Best hyperparameters: {best_hyperparameters}")
print(f"Best validation loss: {study.best_value}")

# Load the best hyperparameters from the JSON file
with open('best_hyperparameters_RoBERTa_justification.json', 'r') as f:
    best_hyperparameters = json.load(f)

# Use the loaded hyperparameters for the final model training and evaluation
batch_size = best_hyperparameters['batch_size']
learning_rate = best_hyperparameters['learning_rate']
dropout_rate = best_hyperparameters['dropout_rate']
weight_decay = best_hyperparameters['weight_decay']
num_epochs = best_hyperparameters['num_epochs']

# Define the model with the best hyperparameters
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=num_labels,
    hidden_dropout_prob=dropout_rate,
    attention_probs_dropout_prob=dropout_rate
)

# Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
loss_fn = torch.nn.CrossEntropyLoss()

# DataLoader with the best batch size
train_loader = create_dataloader(train_dataset, batch_size)
val_loader = create_dataloader(val_dataset, batch_size)
test_loader = create_dataloader(test_dataset, batch_size)

# Training the model with the best hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop with loss tracking and early stopping
best_val_loss = float('inf')
early_stopping_counter = 0
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, device)
    val_loss, _, _ = evaluate(model, val_loader, loss_fn, device)
    
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
        }, 'best_model_checkpoint_expanded.pth')
        print("Validation loss improved. Model and state saved.")
    else:
        early_stopping_counter += 1
        print(f"No improvement in validation loss for {early_stopping_counter} epoch(s).")
        
    if early_stopping_counter >= 10:
        print("Early stopping triggered. Stopping training.")
        break

# Load the best model and state for final evaluation
checkpoint = torch.load('best_model_checkpoint_expanded.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
model.to(device)

# Evaluate on training, validation, and test sets
def display_results(model, data_loader, dataset_name):
    loss, all_labels, all_preds = evaluate(model, data_loader, loss_fn, device)
    accuracy = (np.array(all_labels) == np.array(all_preds)).mean() * 100
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=[f"Class {i}" for i in range(num_labels)], digits=2)
    print(f"\n{dataset_name} Loss: {loss:.4f}")
    print(f"{dataset_name} Accuracy: {accuracy:.2f}%")
    print(f"{dataset_name} Multi-Class Confusion Matrix:")
    print(conf_matrix)
    print(f"{dataset_name} Classification Report:")
    print(class_report)


# Display final results
print("\nFinal Evaluation Results:")
display_results(model, train_loader, "Training")
display_results(model, val_loader, "Validation")
display_results(model, test_loader, "Test")