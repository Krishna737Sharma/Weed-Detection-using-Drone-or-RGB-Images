import os
import torch
import yaml
from torch.utils.data import DataLoader
from src.data_loader import CWFIDDataLoader, WeedDataset
from src.model import EfficientWeedDetector, build_model
from src.train import train_model
from src.evaluate import ModelEvaluator
from src.utils import print_model_summary, plot_training_history, save_training_report
import torchvision.transforms as transforms

def download_cwfid_dataset():
    """Placeholder for downloading CWFID dataset"""
    data_path = 'data/dataset-1.0'
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Please download the CWFID dataset (v3.0.1) manually.")
        print("Download from: https://github.com/cwfid/dataset")
        print("Expected structure: data/dataset-1.0/{images,masks,annotations,train_test_split.yaml}")
        raise FileNotFoundError(f"Dataset directory {data_path} not found")
    print("Dataset already exists.")

def main():
    torch.manual_seed(42)
    
    try:
        print("===== Downloading dataset =====")
        download_cwfid_dataset()
        
        print("\n===== Preparing data =====")
        loader = CWFIDDataLoader(data_path='data/dataset-1.0', use_precomputed_masks=False)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset, test_dataset = loader.get_datasets(transform=transform)
        
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            print("Error: No data loaded! Check dataset paths and structure.")
            print(f"Expected data at: {os.path.abspath('data/dataset-1.0')}")
            return
        
        print(f"Data loaded successfully - Train: {len(train_dataset)} samples, Test: {len(test_dataset)} samples")

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

        print("\n===== Building model =====")
        use_transformer = input("Use Transformer model? (y/n): ").lower().startswith('y')
        
        if use_transformer:
            model = EfficientWeedDetector(num_classes=3)  # 3 classes: background, crop, weed
            print_model_summary(model)
        else:
            model = build_model(input_shape=(3, 512, 512), num_classes=3)  # 3 classes
            print_model_summary(model)

        print("\n===== Training model =====")
        trained_model, history = train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            epochs=10,
            batch_size=4,
            learning_rate=1e-4
        )

        print("\n===== Evaluating model =====")
        X_test, y_test = loader.load_split_data()[2:]
        evaluator = ModelEvaluator(trained_model, X_test, y_test, device='cuda')
        report = evaluator.generate_metrics_report()
        evaluator.visualize_results(num_samples=5)
        
        print("\n===== Evaluation Report =====")
        print(report)

        plot_training_history(history)
        save_training_report(history)

        model_path = 'weed_detector_model.pth'
        torch.save(trained_model.state_dict(), model_path)
        print(f"\nModel saved to {os.path.abspath(model_path)}")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting steps:")
        print("1. Verify dataset exists at: data/dataset-1.0/")
        print("2. Check folder contains: images/, masks/, annotations/, train_test_split.yaml")
        print("3. Ensure all required packages are installed")
        print("4. Check for corrupted image files")

if __name__ == "__main__":
    main()