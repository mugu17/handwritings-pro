import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_training_history(history):
    """Plot training and validation accuracy/loss curves with comprehensive error handling"""
    if not hasattr(history, 'history'):
        print("Error: Input doesn't contain training history")
        return
    
    plt.figure(figsize=(12, 4))
    
    # Plot Accuracy - handles different Keras versions
    plt.subplot(1, 2, 1)
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Train Accuracy')
    elif 'acc' in history.history:
        plt.plot(history.history['acc'], label='Train Accuracy')
    else:
        print("Warning: No accuracy metric found in history")
    
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    elif 'val_acc' in history.history:
        plt.plot(history.history['val_acc'], label='Validation Accuracy')
    
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot Loss - more error resistant
    plt.subplot(1, 2, 2)
    if 'loss' in history.history:
        plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, X_test, y_test, class_names=None):
    """Plot confusion matrix with bulletproof error handling"""
    try:
        # Convert y_test to numpy array if it isn't
        y_test = np.array(y_test)
        
        # Handle different y_test formats
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_true = np.argmax(y_test, axis=1)
        else:
            y_true = y_test.flatten()  # Ensure 1D array
            
        # Get predictions safely
        try:
            y_pred = model.predict(X_test, verbose=0)
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                y_pred = np.argmax(y_pred, axis=1)
            else:
                y_pred = y_pred.flatten()
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return
            
        # Generate confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred)
        except ValueError as e:
            print(f"Confusion matrix error: {str(e)}")
            return
            
        # Handle class names
        if class_names is None:
            class_names = np.unique(np.concatenate((y_true, y_pred)))
        class_names = [str(name) for name in class_names]
        
        # Create plot with size adjustment
        plt.figure(figsize=(max(8, len(class_names)), max(6, len(class_names)*0.7)))
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Adjust label rotation based on label type
        rotation = 45 if any(len(str(name)) > 3 for name in class_names) else 0
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Unexpected error in confusion matrix: {str(e)}")

# Example usage with test data
if __name__ == "__main__":
    try:
        # Mock data for testing
        history = type('', (), {'history': {
            'accuracy': [0.7, 0.8, 0.9],
            'val_accuracy': [0.65, 0.75, 0.85],
            'loss': [0.5, 0.3, 0.2],
            'val_loss': [0.6, 0.4, 0.3]
        }})
        
        class MockModel:
            def predict(self, X, verbose=0):
                return np.random.rand(len(X), 3)  # Mock predictions
        
        X_test = np.random.rand(10, 5)  # 10 samples, 5 features
        y_test = np.random.randint(0, 3, 10)  # 10 labels (0-2)
        
        # Test numerical labels
        print("Testing with numerical labels:")
        plot_training_history(history)
        plot_confusion_matrix(MockModel(), X_test, y_test)
        
        # Test alphabetical labels
        print("\nTesting with alphabetical labels:")
        plot_confusion_matrix(MockModel(), X_test, y_test, 
                             class_names=['Apple', 'Banana', 'Cherry'])
        
    except Exception as e:
        print(f"Example usage failed: {str(e)}")