from data_loaders import openwebtext_loader

OWT_loader = openwebtext_loader.OpenWebTextLoader()
OWT_loader.load("./data/raw")
OWT_loader.verify_files()   