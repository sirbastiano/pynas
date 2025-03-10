import torch
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
import configparser
import os
import time
from datetime import datetime
from .. import classes
# from datasets.L0_thraws_classifier.dataset import SentinelDataset, SentinelDataModule
from datasets.RawClassifier.loader import RawClassifierDataset, RawClassifierDataModule
# from datasets.Phisat2SimulatedData.segmentation_dataset import SegmentationDataset, SegmentationDataModule
# from datasets.wake_classifier.dataset import xAIWakesDataModule
import torch.multiprocessing as mp


mp.set_start_method('fork', force=True)


DataClass = RawClassifierDataset
DataModuleClass = RawClassifierDataModule


SegmentationDataset = None
SegmentationDataModule = None

def compute_fitness_value(parsed_layers, log_learning_rate=None, batch_size=None, is_final=False):
    """
    Computes the fitness value for a given network architecture and hyperparameters in NAS.

    This function constructs and trains a PyTorch Lightning model based on the provided architecture and
    hyperparameters. It uses the appropriate DataModule for data loading and preprocessing. The fitness is 
    calculated based on the test accuracy and F1 score obtained after training the model.

    Parameters:
    parsed_layers (list): A list of dictionaries defining the neural network architecture.
    log_learning_rate (float, optional): The logarithm (base 10) of the learning rate.
    batch_size (int, optional): The batch size for training.
    is_final (bool, optional): If True, run the final evaluation and save the model.

    Returns:
    float: The computed fitness value, a weighted average of accuracy and F1 score for classification, or
           a combination of MSE and IoU for segmentation, penalized by the number of parameters.
    """
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Torch stuff
    seed = config.getint(section='Computation', option='seed')
    pl.seed_everything(seed=seed, workers=True)  # For reproducibility
    torch.set_float32_matmul_precision("medium")  # to make lightning happy
    num_workers = config.getint(section='Computation', option='num_workers')
    accelerator = config.get(section='Computation', option='accelerator')

    # Get model parameters
    log_lr = log_learning_rate if log_learning_rate is not None else config.getfloat(section='Search Space', option='default_log_lr')
    
    lr = 10**log_lr
    bs = batch_size if batch_size is not None else config.getint(section='Search Space', option='default_bs')
    print(f"-----------The batch size of the data to be loaded in the model is: {bs}-----------")
    


    # DATA
    root_dir = config['Dataset']['data_path']
    num_classes = config.getint(section='Dataset', option='num_classes')
    task_type = config.get('Task', 'type_head')
    print(f'Task type in fitness is: {task_type}')

    # Select dataset and data module based on task type
    if task_type == 'classification':
        composed_transform = transforms.Compose([
            transforms.ToTensor(),
            # Add any other transforms if needed
        ])
        
        dataset = DataClass(
            root_dir=root_dir,
            transform=composed_transform,
        )
        dm = DataModuleClass(
            root_dir=root_dir,
            batch_size=round(float(bs)),
            num_workers=num_workers,
            transform=composed_transform,
        )
        
        image, label = dataset[0]
        in_channels = image.shape[0]

    elif task_type == 'segmentation':
        composed_transform = transforms.Compose([
            transforms.ToTensor(),
            # Add any other transforms if needed
        ])
        dataset = SegmentationDataset(
            root_dir=root_dir,
            transform=composed_transform,
        )
        dm = SegmentationDataModule(
            root_dir=root_dir,
            batch_size=round(float(bs)),
            num_workers=num_workers,
            transform=composed_transform,
        )
        
        image, label = dataset[0]
        in_channels = image.shape[0]

    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    print(f"\n\n***\n\n{parsed_layers}\n***\n\n")

    try:
        # MODEL
        if not is_final:
            if task_type == 'classification':
                model = classes.GenericLightningNetwork(
                    parsed_layers=parsed_layers,
                    input_channels=in_channels,
                    num_classes=num_classes,
                    learning_rate=lr,
                )
            elif task_type == 'segmentation':
                model = classes.GenericLightningSegmentationNetwork(
                    parsed_layers=parsed_layers,
                    input_channels=in_channels,
                    num_classes=num_classes,
                    learning_rate=lr,
                )
            
            
           

            # Check the number of parameters
            num_params = sum(p.numel() for p in model.parameters())
            if num_params > 131000000:
                print(f"Skipping architecture, total parameters: {num_params} exceed the threshold of 10M")
                return float('-inf')  # or another value to indicate invalid architecture
            
            print("Running in not final loop")
            trainer = pl.Trainer(
                accelerator=accelerator,
                devices=1,  # Specify the number of GPUs to use
                #strategy='horovod',
                min_epochs=1,
                max_epochs=50,
                fast_dev_run=False,
                check_val_every_n_epoch=51,
                callbacks=[classes.TrainEarlyStopping(monitor='train_loss', mode="min", patience=5)]
            )

            # Training
            training_start_time = time.time()
            trainer.fit(model, dm)
            training_time = time.time() - training_start_time

            trainer.validate(model, dm)

            # Test
            test_start_time = time.time()
            results = trainer.test(model, dm)
            print(f"Results are: {results}")
            
            test_time = time.time() - test_start_time
            
            # Ensure the test dataset is set up
            dm.setup(stage='test')
            
            # Calculate FPS
            if hasattr(dm, 'test_dataset'):
                num_test_samples = len(dm.test_dataset)  # Number of samples in the test set
                fps = num_test_samples / test_time  # Frames per second
            else:
                fps = None
                print("Test dataset is not available.")
     

            # Calculate fitness based on task type
            if task_type == 'classification':
                acc = results[0].get('test_accuracy')
                f1 = results[0].get('test_f1_score')
                mcc = results[0].get('test_mcc')
                fitness = (20 * acc) - (sum(p.numel() for p in model.parameters()) / 1000000)
                iou = None  ##
            elif task_type == 'segmentation':
                mse = results[0].get('test_mse')
                print(f"MSE value is {mse}")
                iou = results[0].get('test_iou')
                print(f"IoU value is {iou}")
                num_param = sum(p.numel() for p in model.parameters())
                print(f"num_param value is {num_param}")
                
                fitness = 10 * (1 / (mse + 1) + iou) - (num_param / 1000000)
                
            model_size = num_params ##

            print(f"Training time: {training_time}")
            print(f"Fitness: {fitness}")
            print("********")
            print("\n" * 4)

            return fitness, iou, fps, model_size ##

        else:
            print("\nFINAL RUN ON OPTIMIZED ARCHITECTURE")

            # MODEL
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            checkpoints_path = config.get(section="Logging", option="checkpoints_dir")
            logger = TensorBoardLogger(save_dir=checkpoints_path, name=f"OptimizedModel_{current_datetime}")
            model_dir = f'Models'
            
            if task_type == 'classification':
                model = classes.GenericLightningNetwork(
                    parsed_layers=parsed_layers,
                    input_channels=in_channels,
                    num_classes=num_classes,
                    learning_rate=lr,
                )
            elif task_type == 'segmentation':
                model = classes.GenericLightningSegmentationNetwork(
                    parsed_layers=parsed_layers,
                    input_channels=in_channels,
                    num_classes=num_classes,
                    learning_rate=lr,
                )
            
            # Check the number of parameters
            num_params = sum(p.numel() for p in model.parameters())
            if num_params > 131000000:
                print(f"Skipping architecture, total parameters: {num_params} exceed the threshold of 10M")
                return float('-inf'), None, None, None  # or another value to indicate invalid architecture
            
            print("Running in final loop")
            trainer = pl.Trainer(
                accelerator=accelerator,
                devices=4,  # Specify the number of GPUs to use
                #strategy='horovod',  # Use Distributed Data Parallel strategy
                min_epochs=1,
                max_epochs=50,
                logger=logger,
                check_val_every_n_epoch=1,
                callbacks=[EarlyStopping(monitor='val_loss', mode="min", patience=5)]
            )

            # Training
            training_start_time = time.time()
            trainer.fit(model, dm)
            training_time = time.time() - training_start_time

            trainer.validate(model, dm)

            # Test
            test_start_time = time.time()
            results = trainer.test(model, dm)
            test_time = time.time() - test_start_time
            
            # Ensure the test dataset is set up
            dm.setup(stage='test')
            
            # Calculate FPS
            if hasattr(dm, 'test_dataset'):
                num_test_samples = len(dm.test_dataset)  # Number of samples in the test set
                fps = num_test_samples / test_time  # Frames per second
            else:
                fps = None
                print("Test dataset is not available.")

            # Calculate fitness based on task type
            if task_type == 'classification':
                acc = results[0].get('test_accuracy')
                f1 = results[0].get('test_f1_score')
                mcc = results[0].get('test_mcc')
                fitness = (20 * acc) - (sum(p.numel() for p in model.parameters()) / 1000000)
                iou = None  ##
      
            elif task_type == 'segmentation':
                mse = results[0].get('test_mse')
                print(f"MSE value is {mse}")
                iou = results[0].get('test_iou')
                print(f"IoU value is {iou}")
                num_param = sum(p.numel() for p in model.parameters())
                print(f"num_param value is {num_param}")
                
                fitness = 10*(1 / (mse + 1) + iou) - (num_param / 1000000)
                   
                
            model_size = num_params    ##

            print("FINAL RUN COMPLETED:")
            print(f"Training time: {training_time}")
            print(f"Fitness: {fitness}")
            print("********")
            
            # 1. Save the model's weights separately in the new directory
            weights_save_path = os.path.join(model_dir, f'OptimizedModelWeights_{current_datetime}.pt')
            torch.save(model.state_dict(), weights_save_path)
            print(f"Saved model weights to {weights_save_path}")

            # 2. Save the model architecture to a text file in the new directory
            architecture_save_path = os.path.join(model_dir, f'OptimizedModelArchitecture_{current_datetime}.txt')
            with open(architecture_save_path, 'w') as f:
                f.write(str(parsed_layers))
            print(f"Saved model architecture to {architecture_save_path}")

            # 3. Save the entire model (architecture + weights) to a single file in the new directory
            full_model_save_path = os.path.join(model_dir, f'OptimizedModelFull_{current_datetime}.pt')
            torch.save(model, full_model_save_path)
            print(f"Saved entire model (architecture + weights) to {full_model_save_path}")


            txt_filename = f'Optimized_Architecture_Final_Run_{current_datetime}.txt'
            txt_filepath = os.path.join(checkpoints_path, txt_filename)
            with open(txt_filepath, 'w') as txt_file:
                txt_file.write(f"For the following architecture:\n{parsed_layers}\n")
                txt_file.write(f"\nTraining time: {training_time}")
                txt_file.write(f"\nFitness: {fitness}")
            print(f"\nFinal run text file saved: {txt_filepath}")

            return fitness, iou, fps, model_size ##

    except ValueError as e:
        print(f"Skipping architecture due to error: {e}")
        return float('-inf'), None, None, None  # or another value to indicate invalid architecture
